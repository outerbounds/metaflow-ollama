from metaflow.cards import Markdown, Table, VegaChart
from metaflow.metaflow_current import current
from datetime import datetime
import threading
import time


from metaflow.exception import MetaflowException
from collections import defaultdict


class CardDecoratorInjector:
    """
    Mixin Useful for injecting @card decorators from other first class Metaflow decorators.
    """

    _first_time_init = defaultdict(dict)

    @classmethod
    def _get_first_time_init_cached_value(cls, step_name, card_id):
        return cls._first_time_init.get(step_name, {}).get(card_id, None)

    @classmethod
    def _set_first_time_init_cached_value(cls, step_name, card_id, value):
        cls._first_time_init[step_name][card_id] = value

    def _card_deco_already_attached(self, step, card_id):
        for decorator in step.decorators:
            if decorator.name == "card":
                if decorator.attributes["id"] and card_id == decorator.attributes["id"]:
                    return True
        return False

    def _get_step(self, flow, step_name):
        for step in flow:
            if step.name == step_name:
                return step
        return None

    def _first_time_init_check(self, step_dag_node, card_id):
        """ """
        return not self._card_deco_already_attached(step_dag_node, card_id)

    def attach_card_decorator(
        self,
        flow,
        step_name,
        card_id,
        card_type,
        refresh_interval=5,
    ):
        """
        This method is called `step_init` in your StepDecorator code since
        this class is used as a Mixin
        """
        from metaflow import decorators as _decorators

        if not all([card_id, card_type]):
            raise MetaflowException(
                "`INJECTED_CARD_ID` and `INJECTED_CARD_TYPE` must be set in the `CardDecoratorInjector` Mixin"
            )

        step_dag_node = self._get_step(flow, step_name)
        if (
            self._get_first_time_init_cached_value(step_name, card_id) is None
        ):  # First check class level setting.
            if self._first_time_init_check(step_dag_node, card_id):
                self._set_first_time_init_cached_value(step_name, card_id, True)
                _decorators._attach_decorators_to_step(
                    step_dag_node,
                    [
                        "card:type=%s,id=%s,refresh_interval=%s"
                        % (card_type, card_id, str(refresh_interval))
                    ],
                )
            else:
                self._set_first_time_init_cached_value(step_name, card_id, False)


class CardRefresher:

    CARD_ID = None

    def on_startup(self, current_card):
        raise NotImplementedError("make_card method must be implemented")

    def on_error(self, current_card, error_message):
        raise NotImplementedError("error_card method must be implemented")

    def on_update(self, current_card, data_object):
        raise NotImplementedError("update_card method must be implemented")

    def sqlite_fetch_func(self, conn):
        raise NotImplementedError("sqlite_fetch_func must be implemented")


class OllamaStatusCard(CardRefresher):
    """
    Real-time status card for Ollama system monitoring.
    Shows circuit breaker state, server health, model status, and recent events.

    Intended to be inherited from in a step decorator like this:
        class OllamaDecorator(StepDecorator, OllamaStatusCard):
    """

    CARD_ID = "ollama_status"

    def __init__(self, refresh_interval=10):
        self.refresh_interval = refresh_interval
        self.status_data = {
            "circuit_breaker": {
                "state": "CLOSED",
                "failure_count": 0,
                "last_failure_time": None,
                "last_open_time": None,
            },
            "server": {
                "status": "Starting",
                "uptime_start": None,
                "restart_attempts": 0,
                "last_health_check": None,
                "health_status": "Unknown",
            },
            "models": {},  # model_name -> {status, pull_time, load_time, etc}
            "performance": {
                "install_time": None,
                "server_startup_time": None,
                "total_initialization_time": None,
            },
            "versions": {
                "ollama_system": "Detecting...",
                "ollama_python": "Detecting...",
            },
            "cache": {
                "policy": "auto",
                "model_status": {},  # model_name -> cache status
            },
            "events": [],  # Recent events log
        }
        self._lock = threading.Lock()
        self._already_rendered = False

    def update_status(self, category, data):
        """Thread-safe method to update status data"""
        with self._lock:
            if category in self.status_data:
                self.status_data[category].update(data)

    def add_event(self, event_type, message, timestamp=None):
        """Add an event to the timeline"""
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            self.status_data["events"].insert(
                0,
                {
                    "type": event_type,  # 'info', 'warning', 'error', 'success'
                    "message": message,
                    "timestamp": timestamp,
                },
            )
            # Keep only last 10 events
            self.status_data["events"] = self.status_data["events"][:10]

    def get_circuit_breaker_emoji(self, state):
        """Get status emoji for circuit breaker state"""
        emoji_map = {"CLOSED": "🟢", "OPEN": "🔴", "HALF_OPEN": "🟡"}
        return emoji_map.get(state, "⚪")

    def get_uptime_string(self, start_time):
        """Calculate uptime string"""
        if not start_time:
            return "Not started"

        uptime = datetime.now() - start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def on_startup(self, current_card):
        """Initialize the card when monitoring starts"""
        current_card.append(Markdown("# 🦙 `@ollama` Status Dashboard"))
        current_card.append(Markdown("_Initializing Ollama system..._"))
        current_card.refresh()

    def render_card_fresh(self, current_card, data):
        """Render the complete card with all status information"""
        self._already_rendered = True
        current_card.clear()

        current_card.append(Markdown("# 🦙 `@ollama` Status Dashboard"))

        versions = data.get("versions", {})
        system_version = versions.get("ollama_system", "Unknown")
        python_version = versions.get("ollama_python", "Unknown")
        current_card.append(
            Markdown(
                f"**System:** `{system_version}` | **Python Client:** `{python_version}`"
            )
        )

        cache_info = data.get("cache", {})
        cache_policy = cache_info.get("policy", "auto")
        current_card.append(Markdown(f"**Cache Policy:** `{cache_policy}`"))

        current_card.append(
            Markdown(f"_Last updated: {datetime.now().strftime('%H:%M:%S')}_")
        )

        cb_data = data["circuit_breaker"]
        cb_emoji = self.get_circuit_breaker_emoji(cb_data["state"])
        cb_status = f"{cb_emoji} **{cb_data['state']}**"
        if cb_data["failure_count"] > 0:
            cb_status += f" (failures: {cb_data['failure_count']})"

        server_data = data["server"]
        uptime = self.get_uptime_string(server_data.get("uptime_start"))
        server_status = f"**{server_data['status']}**"
        if server_data["restart_attempts"] > 0:
            server_status += f" (restarts: {server_data['restart_attempts']})"

        status_table = [
            ["Circuit Breaker", Markdown(cb_status)],
            ["Server Status", Markdown(server_status)],
            ["Server Uptime", Markdown(uptime)],
            [
                "Last Health Check",
                Markdown(server_data.get("health_status", "Unknown")),
            ],
        ]

        current_card.append(Markdown("## System Status"))
        current_card.append(Table(status_table, headers=["Component", "Status"]))

        # Models Status
        if data["models"]:
            current_card.append(Markdown("## Models"))
            model_table = []
            cache_model_status = cache_info.get("model_status", {})

            for model_name, model_info in data["models"].items():
                status = model_info.get("status", "Unknown")
                pull_time = model_info.get("pull_time", "N/A")
                if isinstance(pull_time, (int, float)):
                    pull_time = f"{pull_time:.1f}s"

                cache_status = cache_model_status.get(model_name, "unknown")
                cache_emoji = {
                    "exists": "💾",
                    "missing": "❌",
                    "error": "⚠️",
                    "unknown": "❓",
                }.get(cache_status, "❓")

                size_formatted = model_info.get("size_formatted", "Unknown")
                blob_count = model_info.get("blob_count", "Unknown")
                if blob_count == 0:
                    blob_count = "Unknown"

                model_table.append(
                    [
                        f"{model_name} {cache_emoji}",
                        status,
                        pull_time,
                        size_formatted,
                        str(blob_count),
                    ]
                )

            current_card.append(
                Table(
                    model_table,
                    headers=["Model (Cache)", "Status", "Pull Time", "Size", "Blobs"],
                )
            )

        perf_data = data["performance"]
        if any(v is not None for v in perf_data.values()):
            current_card.append(Markdown("## Performance"))

            init_metrics = []
            shutdown_metrics = []
            other_metrics = []

            for metric, value in perf_data.items():
                if value is not None:
                    display_value = value
                    if isinstance(value, (int, float)):
                        display_value = f"{value:.1f}s"

                    metric_display = metric.replace("_", " ").title()

                    if "shutdown" in metric.lower():
                        shutdown_metrics.append([metric_display, display_value])
                    elif metric in [
                        "install_time",
                        "server_startup_time",
                        "total_initialization_time",
                    ]:
                        init_metrics.append([metric_display, display_value])
                    else:
                        other_metrics.append([metric_display, display_value])

            if init_metrics:
                current_card.append(Markdown("### Initialization"))
                current_card.append(Table(init_metrics, headers=["Metric", "Duration"]))

            if shutdown_metrics:
                current_card.append(Markdown("### Shutdown"))
                current_card.append(
                    Table(shutdown_metrics, headers=["Metric", "Value"])
                )

            if other_metrics:
                current_card.append(Markdown("### Other"))
                current_card.append(Table(other_metrics, headers=["Metric", "Value"]))

        if data["events"]:
            current_card.append(Markdown("## Recent Events"))
            events_table = []
            for event in data["events"][:5]: 
                timestamp = event["timestamp"].strftime("%H:%M:%S")
                event_type = event["type"]
                message = event["message"]

                type_emoji = {
                    "info": "ℹ️",
                    "success": "✅",
                    "warning": "⚠️",
                    "error": "❌",
                }.get(event_type, "ℹ️")

                events_table.append([timestamp, f"{type_emoji} {message}"])

            current_card.append(Table(events_table, headers=["Time", "Event"]))

        current_card.refresh()

    def on_error(self, current_card, error_message):
        """Handle errors in card rendering"""
        if not self._already_rendered:
            current_card.clear()
            current_card.append(Markdown("# 🦙 `@ollama` Status Dashboard"))
            current_card.append(Markdown(f"## ❌ Error: {str(error_message)}"))
            current_card.refresh()

    def on_update(self, current_card, data_object):
        """Update the card with new data"""
        with self._lock:
            current_data = self.status_data.copy()

        if not self._already_rendered:
            self.render_card_fresh(current_card, current_data)
        else:
            # For frequent updates, we could implement incremental updates here
            # For now, just re-render the whole card
            self.render_card_fresh(current_card, current_data)

    def sqlite_fetch_func(self, conn):
        """Required by CardRefresher (which needs a refactor), but we use in-memory data instead"""
        with self._lock:
            return {"status": self.status_data}
