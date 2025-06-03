from metaflow.decorators import StepDecorator
from metaflow import current
import functools
import os
import threading

from .ollama import OllamaManager, OllamaRequestInterceptor
from .status_card import OllamaStatusCard, CardDecoratorInjector

class OllamaDecorator(StepDecorator, CardDecoratorInjector):
    """
    This decorator is used to run Ollama APIs as Metaflow task sidecars.

    User code call
    --------------
    @ollama(
        models=[...],
        ...
    )

    Valid backend options
    ---------------------
    - 'local': Run as a separate process on the local task machine.
    - (TODO) 'managed': Outerbounds hosts and selects compute provider.
    - (TODO) 'remote': Spin up separate instance to serve Ollama models.

    Valid model options
    -------------------
    Any model here https://ollama.com/search, e.g. 'llama3.2', 'llama3.3'

    Parameters
    ----------
    models: list[str]
        List of Ollama containers running models in sidecars.
    backend: str
        Determines where and how to run the Ollama process.
    force_pull: bool
        Whether to run `ollama pull` no matter what, or first check the remote cache in Metaflow datastore for this model key.
    cache_update_policy: str
        Cache update policy: "auto", "force", or "never".
    force_cache_update: bool
        Simple override for "force" cache update policy.
    debug: bool
        Whether to turn on verbose debugging logs.
    circuit_breaker_config: dict
        Configuration for circuit breaker protection. Keys: failure_threshold, recovery_timeout, reset_timeout.
    timeout_config: dict
        Configuration for various operation timeouts. Keys: pull, stop, health_check, install, server_startup.
    """

    name = "ollama"
    defaults = {
        "models": [],
        "backend": "local",
        "force_pull": False,
        "cache_update_policy": "auto",
        "force_cache_update": False,
        "debug": False,
        "circuit_breaker_config": {
            "failure_threshold": 3,
            "recovery_timeout": 60,
            "reset_timeout": 30,
        },
        "timeout_config": {
            "pull": 600, 
            "stop": 30,  
            "health_check": 5,
            "install": 60, 
            "server_startup": 300, 
        },
        "card_refresh_interval": 10,
    }

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        super().step_init(
            flow, graph, step_name, decorators, environment, flow_datastore, logger
        )
        self.flow_datastore_backend = flow_datastore._storage_impl

        # Attach the ollama status card
        self.attach_card_decorator(
            flow,
            step_name,
            "ollama_status",
            "blank",
            refresh_interval=self.attributes["card_refresh_interval"],
        )

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        @functools.wraps(step_func)
        def ollama_wrapper():
            self.ollama_manager = None
            self.request_interceptor = None
            self.status_card = None
            self.card_monitor_thread = None

            try:
                self.status_card = OllamaStatusCard(
                    refresh_interval=self.attributes["card_refresh_interval"]
                )

                def monitor_card():
                    try:
                        self.status_card.on_startup(current.card["ollama_status"])

                        while not getattr(
                            self.card_monitor_thread, "_stop_event", False
                        ):
                            try:
                                self.status_card.on_update(
                                    current.card["ollama_status"], None
                                )
                                import time

                                time.sleep(self.attributes["card_refresh_interval"])
                            except Exception as e:
                                if self.attributes["debug"]:
                                    print(f"[@ollama] Card monitoring error: {e}")
                                break
                    except Exception as e:
                        if self.attributes["debug"]:
                            print(f"[@ollama] Card monitor thread error: {e}")
                        self.status_card.on_error(current.card["ollama_status"], str(e))

                self.card_monitor_thread = threading.Thread(
                    target=monitor_card, daemon=True
                )
                self.card_monitor_thread._stop_event = False
                self.card_monitor_thread.start()

                self.ollama_manager = OllamaManager(
                    models=self.attributes["models"],
                    backend=self.attributes["backend"],
                    flow_datastore_backend=self.flow_datastore_backend,
                    force_pull=self.attributes["force_pull"],
                    cache_update_policy=self.attributes["cache_update_policy"],
                    force_cache_update=self.attributes["force_cache_update"],
                    debug=self.attributes["debug"],
                    circuit_breaker_config=self.attributes["circuit_breaker_config"],
                    timeout_config=self.attributes["timeout_config"],
                    status_card=self.status_card,
                )

                # Monkey-patch key ollama python client interfaces to pass requests through health monitor.
                self.request_interceptor = OllamaRequestInterceptor(
                    self.ollama_manager.circuit_breaker, self.attributes["debug"]
                )
                self.request_interceptor.install_protection()

                if self.attributes["debug"]:
                    print(
                        "[@ollama] OllamaManager initialized and request protection installed"
                    )

            except Exception as e:
                if self.status_card:
                    self.status_card.add_event(
                        "error", f"Initialization failed: {str(e)}"
                    )
                    try:
                        self.status_card.on_error(current.card["ollama_status"], str(e))
                    except:
                        pass
                print(f"[@ollama] Error initializing OllamaManager: {e}")
                raise

            try:
                if self.status_card:
                    self.status_card.add_event("info", "Starting user step function")
                step_func()
                if self.status_card:
                    self.status_card.add_event(
                        "success", "User step function completed successfully"
                    )
            finally:
                # Remove request protection first (before terminating models)
                if self.request_interceptor:
                    self.request_interceptor.remove_protection()
                    if self.attributes["debug"]:
                        print("[@ollama] Request protection removed")

                if self.ollama_manager:
                    self.ollama_manager.terminate_models()

                if self.card_monitor_thread and self.status_card:
                    import time

                    try:
                        self.status_card.on_update(current.card["ollama_status"], None)
                    except Exception as e:
                        if self.attributes["debug"]:
                            print(f"[@ollama] Final card update error: {e}")
                    time.sleep(2)  # Allow final events to be rendered

                if self.card_monitor_thread:
                    self.card_monitor_thread._stop_event = True

                if self.ollama_manager and self.attributes["debug"]:
                    print(
                        f"[@ollama] process statuses: {self.ollama_manager.processes}"
                    )
                    print(
                        f"[@ollama] process runtime stats: {self.ollama_manager.stats}"
                    )
                    print(
                        f"[@ollama] Circuit Breaker status: {self.ollama_manager.circuit_breaker.get_status()}"
                    )

        return ollama_wrapper