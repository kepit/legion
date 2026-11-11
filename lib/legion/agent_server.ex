defmodule Legion.AgentServer do
  @moduledoc """
  GenServer that maintains conversation history for a long-lived agent.

  Holds the message history across multiple turns. Each `call` or `cast`
  appends the user message and runs `Executor` to completion. Requests run
  one at a time, in the order they arrive, each in a worker process linked to
  the agent, so the agent stays free to cancel a running or queued call made
  with a `:ref`, or tell the code it ran; see `Legion.cancel/2` and
  `Legion.get_code/2`.

  When Legion starts an agent in resume or recovery mode, the server restores
  the persisted conversation and executor checkpoint, then continues it during
  startup. Recovery stops the temporary process after that execution finishes.
  """

  use GenServer

  require Logger

  alias Legion.{Eval, Executor, Store, Telemetry}
  alias Legion.RateLimiter
  alias Legion.RateLimiter.ExceededError
  alias Legion.RateLimiter.Policy
  alias Legion.Store.Payload
  alias Legion.Tools.AgentTool
  alias ReqLLM.Message.ContentPart

  @legion_vault_keys ~w(agent_id parent_agent_id agent_module sandbox store rate_limit sub_agents)a

  defstruct [
    :agent_module,
    :messages,
    :config,
    :store,
    :agent_id,
    :persistence_frequency,
    :track_usage,
    usage: nil,
    executor_state: :nonexistent,
    bindings: [],
    # Under `:turn` scope, what a resumed turn goes back to when it ends: the
    # bindings `Legion.eval/3` made outside any turn, as its checkpoint saved them.
    base_bindings: [],
    idle_timer: nil,
    # Requests waiting for the one running in `running`, a worker process.
    queue: :queue.new(),
    running: nil,
    # The code each finished or cancelled request with a `:ref` ran, by ref,
    # as `{code, :ok | :error | :running}`.
    codes: %{}
  ]

  @cancelled_note "This request was cancelled before it finished. " <>
                    "Do not continue it unless you are asked again."

  # Client API

  def start_link(agent_module, opts \\ []) do
    {init_arg, gen_opts} = start_args(agent_module, opts)

    GenServer.start_link(__MODULE__, init_arg, gen_opts)
  end

  @doc false
  def start_monitor(agent_module, opts \\ []) do
    {init_arg, gen_opts} = start_args(agent_module, opts)
    {name, gen_opts} = Keyword.pop!(gen_opts, :name)

    :gen_server.start_monitor(name, __MODULE__, init_arg, gen_opts)
  end

  defp start_args(agent_module, opts) do
    {store, opts} = Keyword.pop(opts, :store)
    {rate_limit_opts, opts} = Keyword.pop(opts, :rate_limit)
    {agent_id, opts} = Keyword.pop(opts, :agent_id)
    {vault, opts} = Keyword.pop(opts, :vault, [])

    store = store || Vault.get(:store) || Application.get_env(:legion, :store)

    if is_nil(store) and not is_nil(agent_id) do
      raise ArgumentError,
            ":agent_id requires a :store - pass one or set `config :legion, :store, MyStore`"
    end

    agent_id =
      cond do
        is_nil(agent_id) ->
          generate_id()

        is_binary(agent_id) and String.valid?(agent_id) ->
          agent_id

        true ->
          raise ArgumentError, ":agent_id must be a valid UTF-8 string, got: #{inspect(agent_id)}"
      end

    persistence_frequency = Store.persistence_frequency(store)
    track_usage = Application.get_env(:legion, :track_usage, true)

    rate_limit = RateLimiter.resolve!(rate_limit_opts)
    check_store!(rate_limit, store, track_usage)
    gen_opts = [name: Legion.AgentIndex.name(agent_id)]
    config = Legion.Agent.resolve_config(agent_module, opts)
    Legion.Agent.warn_unknown_keys(config)
    config = Map.put(config, :rate_limit, rate_limit)

    callers = inherited_callers()

    {{agent_module, config, store, agent_id, persistence_frequency, track_usage, vault, callers},
     gen_opts}
  end

  # Runs in the starting process, and init/1 sets the result in the agent: what a
  # Task does, so a test's Ecto sandbox connection and stubs reach the agent.
  # Ancestors cover a supervisor the test started, which has no callers. One from
  # outside the test breaks the chain, so its agents need an allowance or shared
  # mode. The parent stays first: Vault follows the first caller.
  defp inherited_callers do
    ancestors =
      for ancestor <- Process.get(:"$ancestors", []), pid = GenServer.whereis(ancestor), do: pid

    [self() | Process.get(:"$callers", [])] ++ ancestors
  end

  defp check_store!(%{limiter: limiter, rules: rules}, store, track_usage)
       when not is_nil(limiter) and rules != [] do
    if Code.ensure_loaded?(limiter) and function_exported?(limiter, :check_store!, 3),
      do: limiter.check_store!(store, track_usage, rules)
  end

  defp check_store!(_rate_limit, _store, _track_usage), do: :ok

  def call(agent, message, opts \\ [])

  def call(agent, message, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, [:ref, timeout: :infinity])
    GenServer.call(agent, {:message, message, opts[:ref]}, opts[:timeout])
  end

  def call(agent, message, timeout), do: call(agent, message, timeout: timeout)

  @doc false
  def cancel(agent, ref) when not is_nil(ref) do
    GenServer.call(agent, {:cancel, ref})
  end

  @doc false
  def get_code(agent, ref, opts \\ []) when not is_nil(ref) do
    opts = Keyword.validate!(opts, include_failed: true)
    GenServer.call(agent, {:get_code, ref, opts})
  end

  def cast(agent, message) do
    GenServer.cast(agent, {:message, message})
  end

  @doc false
  # Runs `code` written by an outside model (an MCP host) as one step of this
  # conversation - rate-limited and persisted like a turn, with no LLM request.
  # Returns `{:ok, text}`, `{:error, text}` or `{:cancel, {:rate_limited, violations}}`.
  def eval(agent, code, opts \\ []) do
    {timeout, opts} = Keyword.pop(opts, :timeout, :infinity)
    GenServer.call(agent, {:eval, code, opts}, timeout)
  end

  def get_messages(agent) do
    GenServer.call(agent, :get_messages)
  end

  def get_agent_id(agent) do
    GenServer.call(agent, :get_agent_id)
  end

  # Server callbacks

  @impl true
  def init(
        {agent_module, config, store, agent_id, _frequency, _track_usage, _vault, callers} =
          init_arg
      ) do
    Process.put(:"$callers", callers)
    stored = store && store.get(agent_id)

    case stored do
      {:ok, %Payload{agent_module: stored_module}}
      when not is_nil(stored_module) and stored_module != agent_module ->
        {:error, {:agent_module_mismatch, stored_module}}

      _stored ->
        case enforce_start(%{agent_module: agent_module, agent_id: agent_id, config: config}) do
          :ok -> start(init_arg, stored)
          {:rate_limited, violations} -> {:error, {:rate_limited, violations}}
        end
    end
  end

  # A start is checked against `:max_agents` only, so it counts the agent from
  # its start. Every turn checks the other limits anyway, and a supervisor
  # restarts a crashed agent through this same start: denying it on spent
  # tokens would fail every restart until the supervisor gives up and takes
  # its other agents down. The restarted agent's id is already counted, so
  # `:max_agents` lets it through. Resumed and recovered runs finish work that
  # was already allowed, so they are not checked again.
  defp enforce_start(%{config: config} = state) do
    case Map.get(config, :start_mode, :normal) do
      :normal ->
        case agent_limit_only(config.rate_limit.rules) do
          [] -> :ok
          rules -> enforce_rate_limit(put_in(state.config.rate_limit.rules, rules))
        end

      _resume_or_recover ->
        :ok
    end
  end

  defp agent_limit_only(rules) do
    for %{policy: %Policy{max_agents: max_agents} = policy} = rule <- List.wrap(rules),
        not is_nil(max_agents),
        do: %{rule | policy: %Policy{window_ms: policy.window_ms, max_agents: max_agents}}
  end

  defp start(
         {agent_module, config, store, agent_id, persistence_frequency, track_usage, vault,
          _callers},
         stored
       ) do
    parent_agent_id = Vault.get(:agent_id)
    mode = Map.get(config, :start_mode, :normal)

    for {key, value} <- vault, do: Vault.unsafe_put(key, value)
    Vault.unsafe_put(:agent_id, agent_id)
    Vault.unsafe_put(:parent_agent_id, parent_agent_id)
    Vault.unsafe_put(:agent_module, agent_module)
    Vault.unsafe_put(:sandbox, config.sandbox)
    if store, do: Vault.unsafe_put(:store, store)

    Legion.Agent.seed_tool_configs(agent_module)
    Vault.unsafe_put(:rate_limit, config.rate_limit)

    Vault.unsafe_put(:sub_agents, %{
      max: config.max_sub_agents,
      idle_timeout: config.sub_agent_idle_timeout
    })

    system_prompt = Legion.AgentPrompt.system_prompt(agent_module, config)

    Telemetry.emit(
      [:legion, :agent, :started],
      %{system_time: NaiveDateTime.utc_now()},
      %{agent: agent_module}
    )

    {saved_messages, saved_bindings, saved_base, saved_executor_state, saved_usage} =
      case stored do
        {:ok,
         %Payload{
           conversation_state:
             %{
               messages: messages,
               bindings: bindings,
               executor_state: executor_state
             } = conversation_state,
           usage: usage
         }} ->
          {messages, bindings, Map.get(conversation_state, :base_bindings, []), executor_state,
           if(track_usage, do: usage || [], else: nil)}

        _no_state ->
          {[], [], [], :nonexistent, if(track_usage, do: [], else: nil)}
      end

    saved_bindings =
      restored_bindings(saved_bindings, saved_base, mode, saved_executor_state, config)

    state = %__MODULE__{
      agent_module: agent_module,
      messages: [Executor.message(:system, system_prompt) | saved_messages],
      config: config,
      store: store,
      agent_id: agent_id,
      persistence_frequency: persistence_frequency,
      bindings: saved_bindings,
      base_bindings: saved_base,
      executor_state: saved_executor_state,
      track_usage: track_usage,
      usage: saved_usage
    }

    {:ok,
     persist(state,
       agent_module: state.agent_module,
       parent_agent_id: parent_agent_id,
       started_at: NaiveDateTime.utc_now(),
       usage: state.usage
     ), {:continue, %{start_mode: mode, executor_state: saved_executor_state}}}
  end

  # A normal start over a checkpoint abandons that turn, so it keeps what the
  # turn would have gone back to. A resume or recovery finishes the turn, with
  # the bindings its checkpoint saved.
  defp restored_bindings(bindings, base, :normal, executor_state, config)
       when executor_state != :nonexistent do
    case Map.get(config, :binding_scope, :turn) do
      :conversation -> bindings
      :turn -> base
      :iteration -> []
    end
  end

  defp restored_bindings(bindings, _base, _mode, _executor_state, _config), do: bindings

  @impl true
  def handle_continue(%{start_mode: :normal}, state), do: {:noreply, reset_idle(state)}

  @impl true
  def handle_continue(%{start_mode: :resume, executor_state: executor_state}, state) do
    if unfinished_turn?(%{messages: state.messages, executor_state: executor_state}) do
      {_reply, state} = do_run(state, executor_state)
      {:noreply, reset_idle(state)}
    else
      {:noreply, reset_idle(state)}
    end
  end

  @impl true
  def handle_continue(%{start_mode: :recover, executor_state: executor_state}, state) do
    {_reply, state} = do_run(state, executor_state)
    {:stop, :normal, state}
  end

  @doc false
  # Whether a conversation state stopped mid-turn: behind a checkpoint, or on
  # a prompt with nothing after it. Resume and recovery finish only such a
  # turn. A `Legion.eval/3` step, an MCP call, leaves neither, and finishing
  # it would run the agent's own model over a conversation an outside model
  # drives; nor does a turn that ended on its eval result.
  def unfinished_turn?(%{executor_state: executor_state}) when executor_state != :nonexistent,
    do: true

  def unfinished_turn?(%{messages: messages}), do: match?(%{type: :user}, List.last(messages))
  def unfinished_turn?(_conversation_state), do: false

  @impl true
  def terminate(_reason, state) do
    # A worker is linked, but a :normal stop does not take it down.
    if state.running do
      Process.unlink(state.running.worker)
      Process.exit(state.running.worker, :kill)
    end

    Telemetry.emit(
      [:legion, :agent, :stopped],
      %{system_time: NaiveDateTime.utc_now()},
      %{agent: state.agent_module}
    )
  end

  @impl true
  def handle_call(:get_messages, from, state) do
    {:noreply, enqueue(state, %{kind: :get_messages, from: from, ref: nil})}
  end

  @impl true
  def handle_call(:get_agent_id, _from, state) do
    {:reply, state.agent_id, reset_idle(state)}
  end

  @impl true
  def handle_call({:message, message, ref}, from, state) do
    {:noreply, enqueue(state, %{kind: :message, message: message, from: from, ref: ref})}
  end

  @impl true
  def handle_call({:eval, code, opts}, from, state) do
    {:noreply, enqueue(state, %{kind: :eval, code: code, opts: opts, from: from, ref: nil})}
  end

  @impl true
  def handle_call({:cancel, ref}, _from, state) do
    {cancelled, kept} = state.queue |> :queue.to_list() |> Enum.split_with(&(&1.ref == ref))
    for job <- cancelled, do: reply(job, {:cancel, :cancelled})
    state = %{state | queue: :queue.from_list(kept)}

    {running_cancelled?, state} =
      case state.running do
        %{job: %{ref: ^ref}} -> {true, state |> cancel_running() |> run_next()}
        _other -> {false, state}
      end

    reply = if running_cancelled? or cancelled != [], do: :ok, else: {:error, :not_found}
    {:reply, reply, state}
  end

  # Answered at once, not queued: the latest request with `ref` is a waiting
  # one, which ran nothing yet, the running one, or a finished one. Code still
  # running, or cut short by a cancel, has not failed.
  @impl true
  def handle_call({:get_code, ref, opts}, _from, state) do
    codes =
      cond do
        Enum.any?(:queue.to_list(state.queue), &(&1.ref == ref)) -> {:ok, []}
        match?(%{job: %{ref: ^ref}}, state.running) -> {:ok, state.running.codes}
        Map.has_key?(state.codes, ref) -> {:ok, Map.fetch!(state.codes, ref)}
        true -> {:error, :not_found}
      end

    reply =
      with {:ok, codes} <- codes do
        {:ok, for({code, status} <- codes, opts[:include_failed] or status != :error, do: code)}
      end

    {:reply, reply, state}
  end

  @impl true
  def handle_cast({:message, message}, state) do
    {:noreply, enqueue(state, %{kind: :message, message: message, from: nil, ref: nil})}
  end

  # A tag no longer in the state is a timer a later call replaced. A busy
  # agent is not idle; it starts the wait over once its queue is empty.
  @impl true
  def handle_info({:idle_timeout, tag}, %{idle_timer: {tag, _timer}, running: nil} = state),
    do: {:stop, :normal, state}

  def handle_info({:job_done, tag, {reply, worker_state}}, %{running: %{tag: tag}} = state) do
    Process.demonitor(state.running.monitor, [:flush])
    reply(state.running.job, reply)

    state = %{
      state
      | running: nil,
        codes: keep_codes(state.codes, state.running),
        messages: worker_state.messages,
        bindings: worker_state.bindings,
        usage: worker_state.usage
    }

    {:noreply, run_next(state)}
  end

  def handle_info({:job_progress, tag, event}, %{running: %{tag: tag} = running} = state) do
    {:noreply, %{state | running: progress(running, event)}}
  end

  # Results and progress of a cancelled worker carry a tag no longer running.
  def handle_info(_message, state), do: {:noreply, state}

  defp progress(running, {:checkpoint, checkpoint}),
    do: %{running | checkpoint: checkpoint, turn_usage: checkpoint.turn_usage}

  defp progress(running, {:usage, turn_usage}), do: %{running | turn_usage: turn_usage}
  defp progress(running, {:code, {:started, code}}),
    do: %{running | codes: running.codes ++ [{code, :running}]}

  defp progress(running, {:code, {:finished, status}}),
    do: %{running | codes: List.update_at(running.codes, -1, fn {code, _} -> {code, status} end)}

  defp keep_codes(codes, %{job: %{ref: nil}}), do: codes
  defp keep_codes(codes, %{job: %{ref: ref}, codes: ran}), do: Map.put(codes, ref, ran)

  defp enqueue(state, job), do: run_next(%{state | queue: :queue.in(job, state.queue)})

  defp run_next(%{running: nil} = state) do
    case :queue.out(state.queue) do
      {:empty, _queue} -> reset_idle(state)
      {{:value, job}, queue} -> start_job(job, %{state | queue: queue})
    end
  end

  defp run_next(state), do: state

  defp start_job(%{kind: :get_messages} = job, state) do
    reply(job, state.messages)
    run_next(state)
  end

  # A call waits while the agent is busy. One whose caller gave up and died
  # meanwhile, as an MCP request does when it times out, is skipped: run now,
  # it would act for nobody, and the caller's retry would run it again.
  defp start_job(%{kind: :eval, from: {caller, _tag}} = job, state) do
    if node(caller) == node() and not Process.alive?(caller) do
      run_next(state)
    else
      spawn_job(job, state, fn state, _report -> handle_eval(job.code, job.opts, state) end)
    end
  end

  defp start_job(%{kind: :message} = job, state) do
    case begin_turn(job.message, state) do
      {:ok, state} ->
        spawn_job(job, state, &do_run(&1, :nonexistent, &2))

      {:rate_limited, violations} ->
        reply(job, {:cancel, {:rate_limited, violations}})
        run_next(state)
    end
  end

  # The worker gets the agent's vault and callers, so tools, stubs and an
  # Ecto sandbox connection reach it as they reach the agent.
  defp spawn_job(job, state, fun) do
    agent = self()
    tag = make_ref()
    vault = Vault.vault(propagate_vault: :none)
    callers = [agent | Process.get(:"$callers", [])]
    report = fn event -> send(agent, {:job_progress, tag, event}) end

    worker =
      spawn_link(fn ->
        Process.put(:"$callers", callers)
        Process.put(:__vault__, vault)
        send(agent, {:job_done, tag, fun.(state, report)})
      end)

    running = %{
      job: job,
      tag: tag,
      worker: worker,
      monitor: Process.monitor(worker),
      started_at: System.monotonic_time(),
      turn_state: state,
      sub_agents_before: if(job.kind == :message, do: turn_sub_agents(state)),
      checkpoint: nil,
      turn_usage: [],
      codes: []
    }

    %{state | running: running}
  end

  defp reply(%{from: nil}, _reply), do: :ok
  defp reply(%{from: from}, reply), do: GenServer.reply(from, reply)

  # Unlinked first, so the kill does not reach the agent; the worker is gone
  # before anything is saved, so none of its saves land after the agent's.
  defp cancel_running(%{running: running} = state) do
    Process.unlink(running.worker)
    Process.exit(running.worker, :kill)

    receive do
      {:DOWN, ref, :process, _pid, _reason} when ref == running.monitor -> :ok
    end

    running = drain_progress(running)
    reply(running.job, {:cancel, :cancelled})
    state = %{state | running: nil, codes: keep_codes(state.codes, running)}
    finish_cancelled_turn(state, running)
  end

  # What the worker sent before it died is already in the mailbox, ahead of
  # its :DOWN, so no code or usage it reported is lost.
  defp drain_progress(%{tag: tag} = running) do
    receive do
      {:job_progress, ^tag, event} -> running |> progress(event) |> drain_progress()
    after
      0 -> running
    end
  end

  # Keeps the prompt and the steps the last checkpoint reached, and tells the
  # model the request was dropped, so neither it nor `Legion.resume/2` picks
  # it up again. The usage of every request already answered is kept too;
  # one whose message was not kept names none.
  defp finish_cancelled_turn(state, %{turn_state: turn} = running) do
    {messages, bindings} =
      case running.checkpoint do
        nil -> {turn.messages, turn.bindings}
        checkpoint -> {checkpoint.messages, checkpoint.bindings}
      end

    conversation_scope? = Map.get(state.config, :binding_scope, :turn) == :conversation

    if running.sub_agents_before,
      do: AgentTool.stop_running(state.agent_id, running.sub_agents_before)

    kept_bindings = if conversation_scope?, do: bindings, else: turn.bindings
    kept_count = length(messages)

    turn_usage =
      Enum.map(running.turn_usage, fn
        %{"message_index" => index} = usage when is_integer(index) and index >= kept_count ->
          %{usage | "message_index" => nil}

        usage ->
          usage
      end)

    usage = persisted_usage(state, turn_usage)

    state = %{
      state
      | messages: messages ++ [Executor.message(:error, @cancelled_note)],
        bindings: kept_bindings,
        usage: usage
    }

    state = persist(state, [:conversation_state, status: :idle, usage: usage])

    Telemetry.emit(
      [:legion, :agent, :message, :stop],
      %{duration: System.monotonic_time() - running.started_at},
      %{
        agent: state.agent_module,
        message: turn.messages |> List.last() |> Map.get(:content),
        iterations: count_assistant(messages) - count_assistant(turn.messages),
        status: :cancel,
        result: :cancelled,
        bindings: kept_bindings
      }
    )

    state
  end

  defp count_assistant(messages), do: Enum.count(messages, &(&1[:role] == "assistant"))

  # A tagged timer rather than a GenServer timeout: that one is reset by any
  # message, not only calls, and fired by a plain `:timeout` from anyone.
  defp reset_idle(state) do
    case Map.get(state.config, :idle_timeout, :infinity) do
      :infinity ->
        state

      milliseconds ->
        if state.idle_timer, do: Process.cancel_timer(elem(state.idle_timer, 1))
        tag = make_ref()
        timer = Process.send_after(self(), {:idle_timeout, tag}, milliseconds)
        %{state | idle_timer: {tag, timer}}
    end
  end

  @doc """
  Normalizes `message`, appends it to the conversation, and runs the executor
  to completion. Returns `{{status, value}, new_state}`.

  Accepted `message` shapes:
    - `binary` - passed verbatim as the user message
    - `{:image, data, media_type}` - single inline image from binary data
    - `{:image_url, url}` - single image from a URL
    - `{:multipart, [ContentPart.t()]}` - mixed text/image/file content; build
      parts with `ReqLLM.Message.ContentPart.text/1`, `image/2`, `image_url/1`,
      `file/3`
    - anything else - rendered via `inspect/2`
  """
  def handle_message(message, state) do
    case begin_turn(message, state) do
      {:ok, state} -> do_run(state)
      {:rate_limited, violations} -> {{:cancel, {:rate_limited, violations}}, state}
    end
  end

  defp begin_turn(message, state) do
    case enforce_rate_limit(state) do
      :ok ->
        content = stringify(message, state.config[:max_message_length])

        # Persist the user message before the turn runs so store-backed views
        # (e.g. the legion_web database source) show it without waiting for the
        # response.
        state =
          state
          |> Map.update!(:messages, &(&1 ++ [Executor.message(:user, content)]))
          |> persist([
            :conversation_state,
            status: :running
          ])

        {:ok, state}

      {:rate_limited, violations} ->
        {:rate_limited, violations}
    end
  end

  defp enforce_rate_limit(%{config: %{rate_limit: %{limiter: limiter, rules: rules}}} = state)
       when not is_nil(limiter) and is_list(rules) do
    :ok = limiter.enforce!(state.agent_id, rules)
  rescue
    error in ExceededError ->
      Telemetry.emit(
        [:legion, :rate_limit, :exceeded],
        %{system_time: NaiveDateTime.utc_now()},
        %{
          agent: state.agent_module,
          agent_id: state.agent_id,
          identity: error.identity,
          policy: error.policy,
          usage: error.usage,
          violations: error.violations
        }
      )

      {:rate_limited, error.violations}
  end

  defp enforce_rate_limit(_state), do: :ok

  # The step is saved once, after it ran, and never as `status: :running`:
  # `Legion.Recovery` would resume a running row through the LLM loop, which a
  # conversation driven by an outside model does not have. Only the rate
  # limiter marks the row running, on enforcing `:max_running_agents`, and
  # the save below clears it. There are no turns here either, so bindings live
  # on unless the scope is `:iteration`. Usage records the evaluation, not
  # tokens: that is what `:max_evals` counts. A step that cannot be saved is
  # answered as an error and forgotten, so the conversation on record and the
  # one in memory stay the same; only its usage entry is kept.
  defp handle_eval(code, opts, state) do
    case eval_refusal(code, opts, state) || enforce_rate_limit(state) do
      :ok ->
        action = %{"action" => "eval_and_continue", "code" => code}
        action_message = Executor.message(:assistant, Jason.encode!(action))

        {reply, result_message, bindings} =
          with_call_vault(opts, state.agent_module, fn -> run_eval(code, state) end)

        entry = %{
          "at" => System.system_time(:millisecond),
          "evals" => 1,
          "message_index" => length(state.messages) - 1
        }

        messages = state.messages ++ [action_message, result_message]
        usage = if state.track_usage, do: state.usage ++ [entry]
        new_state = %{state | messages: messages, bindings: bindings, usage: usage}

        case save(new_state, [:conversation_state, status: :idle, usage: usage]) do
          :ok ->
            {reply, new_state}

          # The evaluation still counts: its entry, pointing at no message,
          # is kept for the next save that succeeds. The limiter may have
          # marked the row running; a status-only save frees that slot if
          # the store answers again.
          :error ->
            _ = save(state, status: :idle)
            usage = if state.track_usage, do: state.usage ++ [%{entry | "message_index" => nil}]

            {{:error,
              "The code ran, but the step could not be saved. Its effects stand; its variables were discarded and do not exist."},
             %{state | usage: usage}}
        end

      {:rate_limited, violations} ->
        {{:cancel, {:rate_limited, violations}}, state}

      {:refused, reason} ->
        {{:error, reason}, state}
    end
  end

  # `:require_agent` and `:require_sandbox` are the caller's conditions on the
  # agent it reached: `Legion.MCP.Server` finds named agents with
  # `Legion.lookup/1`, and one started elsewhere may be any agent, on any sandbox.
  defp eval_refusal(code, opts, %{agent_module: agent_module, config: config}) do
    required_agent = Keyword.get(opts, :require_agent, agent_module)
    required = Keyword.get(opts, :require_sandbox, config.sandbox)
    max_length = config.max_message_length

    evaluates? =
      Enum.any?(agent_module.action_types(), &(&1 in ~w(eval_and_continue eval_and_complete)))

    cond do
      not evaluates? ->
        {:refused,
         "#{inspect(agent_module)} runs no code: its action_types/0 allow no evaluation"}

      required_agent != agent_module ->
        {:refused,
         "This call requires #{inspect(required_agent)}; " <>
           "the agent it reached is #{inspect(agent_module)}"}

      required != config.sandbox ->
        {:refused,
         "This call requires #{inspect(required)}; " <>
           "#{inspect(agent_module)} runs #{inspect(config.sandbox)}"}

      # The code is kept as a message of the conversation, so it is held to
      # the same limit as any other.
      is_integer(max_length) and byte_size(code) > max_length ->
        {:refused,
         "The code is #{byte_size(code)} bytes, over the #{max_length} byte limit; " <>
           "send less at a time"}

      not String.valid?(code) ->
        {:refused, "The code is not valid UTF-8"}

      true ->
        nil
    end
  end

  # A key one caller passed never reaches the next call or a later turn, and no
  # caller replaces the agent's store or identity.
  defp with_call_vault(opts, agent_module, fun) do
    saved = Vault.vault(propagate_vault: :none)

    opts
    |> Keyword.get(:vault, [])
    |> Keyword.drop(@legion_vault_keys)
    |> Keyword.put(:excluded_tools, excluded_tools(opts, agent_module))
    |> Keyword.put(:current_action, "eval_and_continue")
    |> Vault.unsafe_merge()

    try do
      fun.()
    after
      # Vault 0.2.1 cannot drop keys; `Vault.unsafe_replace/1` is in the next
      # release, until then this writes its process dictionary key directly.
      Process.put(:__vault__, saved)
    end
  end

  defp excluded_tools(opts, agent_module) do
    case Keyword.get(opts, :exclude_tools, []) do
      excluded? when is_function(excluded?, 1) -> Enum.filter(agent_module.tools(), excluded?)
      excluded -> excluded
    end
  end

  defp run_eval(code, %{config: config} = state) do
    case Eval.run(state.agent_module, code, config, state.bindings) do
      {:ok, {value, bindings}} ->
        bindings = if config.binding_scope == :iteration, do: [], else: bindings
        text = Eval.format_result(value, bindings, config)
        {{:ok, text}, Executor.message(:eval_result, text), bindings}

      {:error, error} ->
        text =
          error |> Eval.format_error() |> Executor.truncate_content(config.max_message_length)

        {{:error, text}, Executor.message(:error, text), state.bindings}
    end
  end

  # `report`, given when a worker runs the turn, tells the agent how far the
  # turn got, so a cancelled one keeps its finished steps and their usage.
  defp do_run(state, executor_state \\ :nonexistent, report \\ nil) do
    conversation_scope? = Map.get(state.config, :binding_scope, :turn) == :conversation
    # A resumed turn keeps the bindings its checkpoint saved, whatever the scope -
    # they belong to the turn being finished, not to a new one.
    resuming? = executor_state != :nonexistent

    # What the turn goes back to when it ends, unless bindings outlive turns:
    # a new turn starts from it, a resumed turn's checkpoint saved it.
    base = if resuming?, do: state.base_bindings, else: state.bindings
    step? = state.persistence_frequency == :step

    checkpoint =
      if step? or report do
        fn checkpoint ->
          if step? do
            usage = persisted_usage(state, checkpoint.turn_usage)
            checkpoint = Map.put(checkpoint, :base_bindings, base)
            persist(state, [{:conversation_state, checkpoint}, usage: usage])
          end

          if report, do: report.({:checkpoint, checkpoint})
          :ok
        end
      end

    executor_config =
      state.config
      |> Map.put(:checkpoint, checkpoint)
      |> Map.put(:report_usage, report && fn turn_usage -> report.({:usage, turn_usage}) end)
      |> Map.put(:report_code, report && fn code -> report.({:code, code}) end)

    # Unless bindings outlive the turn, the ids of the sub-agents it starts
    # go with them at its end, and so do the sub-agents.
    sub_agents_before = turn_sub_agents(state)

    {status, value, final_messages, final_bindings, turn_usage} =
      Telemetry.span(
        [:legion, :agent, :message],
        %{agent: state.agent_module, message: state.messages |> List.last() |> Map.get(:content)},
        fn ->
          messages = state.messages
          prev_count = Enum.count(messages, &(&1[:role] == "assistant"))

          {status, value, messages, bindings, _turn_usage} =
            result =
            Executor.run(
              state.agent_module,
              messages,
              executor_config,
              state.bindings,
              executor_state
            )

          iterations = Enum.count(messages, &(&1[:role] == "assistant")) - prev_count

          {result,
           %{
             iterations: iterations,
             status: status,
             result: value,
             bindings: bindings
           }}
        end
      )

    if sub_agents_before, do: AgentTool.stop_running(state.agent_id, sub_agents_before)

    kept_bindings = if conversation_scope?, do: final_bindings, else: base

    usage = persisted_usage(state, turn_usage)

    state =
      %{
        state
        | messages: final_messages,
          bindings: kept_bindings,
          usage: usage
      }

    fields = [:conversation_state, status: :idle, usage: usage]
    state = persist(state, fields)

    {{status, value}, state}
  end

  defp turn_sub_agents(state) do
    if Map.get(state.config, :binding_scope, :turn) != :conversation and
         AgentTool in state.agent_module.tools(),
       do: AgentTool.running(state.agent_id)
  end

  defp persist(state, fields) do
    :ok = save(state, fields)
    state
  end

  defp save(%{store: nil}, _fields), do: :ok
  defp save(state, fields), do: state.store.save(payload(state, fields))

  defp payload(state, fields) do
    Enum.reduce(fields, %Payload{agent_id: state.agent_id}, fn
      :conversation_state, payload ->
        %{payload | conversation_state: persisted_conversation_state(state)}

      {:conversation_state, checkpoint}, payload ->
        %{payload | conversation_state: persisted_conversation_state(checkpoint)}

      {field, value}, payload
      when field in [:agent_module, :parent_agent_id, :status, :started_at, :usage] ->
        Map.put(payload, field, value)

      unknown, _payload ->
        raise ArgumentError, "unsupported persistence field: #{inspect(unknown)}"
    end)
  end

  defp persisted_conversation_state(%__MODULE__{} = state) do
    [%{role: "system"} | messages] = state.messages

    %{messages: messages, bindings: state.bindings, executor_state: :nonexistent}
  end

  defp persisted_conversation_state(%{
         messages: [%{role: "system"} | messages],
         bindings: bindings,
         base_bindings: base_bindings,
         executor_state: executor_state
       }) do
    %{
      messages: messages,
      bindings: bindings,
      base_bindings: base_bindings,
      executor_state: executor_state
    }
  end

  defp persisted_usage(state, turn_usage) do
    if state.track_usage, do: state.usage ++ Enum.map(turn_usage, &stored_usage/1)
  end

  # The executor indexes the list it was given, which starts with the system
  # prompt. persisted_conversation_state/1 drops that prompt, so stored
  # entries point one earlier.
  defp stored_usage(%{"message_index" => index} = usage) when is_integer(index),
    do: %{usage | "message_index" => index - 1}

  defp stored_usage(usage), do: usage

  defp generate_id, do: Base.url_encode64(:crypto.strong_rand_bytes(16), padding: false)

  defp stringify(message, max_length) when is_binary(message),
    do: Executor.truncate_content(message, max_length)

  defp stringify({:image, data, media_type}, _max_length)
       when is_binary(data) and is_binary(media_type) do
    [ContentPart.image(data, media_type)]
  end

  defp stringify({:image_url, url}, _max_length) when is_binary(url) do
    [ContentPart.image_url(url)]
  end

  defp stringify({:multipart, parts}, max_length) when is_list(parts) do
    Enum.map(parts, &truncate_text_part(&1, max_length))
  end

  defp stringify(message, max_length) do
    message
    |> inspect(limit: :infinity)
    |> Executor.truncate_content(max_length)
  end

  # Only text parts can be truncated safely - cutting image bytes or a URL
  # corrupts them.
  defp truncate_text_part(%ContentPart{type: :text, text: text} = part, max_length)
       when is_binary(text) do
    %{part | text: Executor.truncate_content(text, max_length)}
  end

  defp truncate_text_part(part, _max_length), do: part
end
