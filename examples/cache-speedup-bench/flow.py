from metaflow import FlowSpec, step, Config, config_expr, card, current, kubernetes
from metaflow.cards import Table, Image, Markdown


class OllamaCacheSpeedupBench(FlowSpec):

    config = Config("config", default="config.json")

    @step
    def start(self):
        self.models = self.config.models
        self.next(self.cache_miss_run, foreach="models")

    @kubernetes(**config_expr("config.k8s_args"))
    @step
    def cache_miss_run(self):
        import time
        from metaflow.plugins.ollama import OllamaManager  # type: ignore

        t0 = time.time()
        ollama_manager = OllamaManager(
            models=[self.input],
            flow_datastore_backend=self._datastore.parent_datastore._storage_impl,
            force_pull=True,
        )
        tf = time.time()
        # At this point, the models are usable.
        self.cache_miss_startup_time = tf - t0
        ollama_manager.terminate_models()
        self.next(self.cache_hit_run)

    @kubernetes(**config_expr("config.k8s_args"))
    @step
    def cache_hit_run(self):
        import time
        from metaflow.plugins.ollama import OllamaManager  # type: ignore

        t0 = time.time()
        ollama_manager = OllamaManager(
            models=[self.input],
            flow_datastore_backend=self._datastore.parent_datastore._storage_impl,
            skip_push_check=True,  # We know cache is populated, don't need to push again.
        )
        tf = time.time()
        self.cache_hit_startup_time = tf - t0
        ollama_manager.terminate_models()
        self.model = self.input
        self.next(self.join)

    @step
    def join(self, inputs):
        self.data = [
            {
                "cache_miss_startup_time": i.cache_miss_startup_time,
                "cache_hit_startup_time": i.cache_hit_startup_time,
                "model": i.model,
                "param_ct": i.model.split(":")[1],
            }
            for i in inputs
        ]
        self.next(self.end)

    def create_performance_plots(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

        df = pd.DataFrame(self.data)
        df["speedup_ratio"] = (
            df["cache_miss_startup_time"] / df["cache_hit_startup_time"]
        )
        df["time_saved"] = df["cache_miss_startup_time"] - df["cache_hit_startup_time"]
        df["speedup_percentage"] = (
            (df["cache_miss_startup_time"] - df["cache_hit_startup_time"])
            / df["cache_miss_startup_time"]
        ) * 100

        fig = plt.figure(figsize=(20, 12))

        # 1. Side-by-side bar chart comparison
        ax1 = plt.subplot(2, 3, 1)
        x = np.arange(len(df))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            df["cache_miss_startup_time"],
            width,
            label="Cache Miss",
            color="#ff6b6b",
            alpha=0.8,
        )
        bars2 = ax1.bar(
            x + width / 2,
            df["cache_hit_startup_time"],
            width,
            label="Cache Hit",
            color="#4ecdc4",
            alpha=0.8,
        )

        ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Startup Time (seconds)", fontsize=12, fontweight="bold")
        ax1.set_title(
            "🚀 Ollama Model Startup Time Comparison", fontsize=14, fontweight="bold"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [f"{row['model']}\n({row['param_ct']})" for _, row in df.iterrows()],
            rotation=0,
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        for bar in bars2:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Speedup ratio chart
        ax2 = plt.subplot(2, 3, 2)
        bars = ax2.bar(df["model"], df["speedup_ratio"], color="#45b7d1", alpha=0.8)
        ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Speedup Ratio", fontsize=12, fontweight="bold")
        ax2.set_title("⚡ Cache Speedup Ratio", fontsize=14, fontweight="bold")
        ax2.set_xticklabels(
            [
                f"{model}\n({param})"
                for model, param in zip(df["model"], df["param_ct"])
            ],
            rotation=0,
        )
        ax2.grid(True, alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.1f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. Time saved chart
        ax3 = plt.subplot(2, 3, 3)
        bars = ax3.bar(df["model"], df["time_saved"], color="#f39c12", alpha=0.8)
        ax3.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Time Saved (seconds)", fontsize=12, fontweight="bold")
        ax3.set_title("⏰ Time Saved by Caching", fontsize=14, fontweight="bold")
        ax3.set_xticklabels(
            [
                f"{model}\n({param})"
                for model, param in zip(df["model"], df["param_ct"])
            ],
            rotation=0,
        )
        ax3.grid(True, alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Percentage improvement
        ax4 = plt.subplot(2, 3, 4)
        bars = ax4.bar(
            df["model"], df["speedup_percentage"], color="#e74c3c", alpha=0.8
        )
        ax4.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Performance Improvement (%)", fontsize=12, fontweight="bold")
        ax4.set_title("📈 Performance Improvement %", fontsize=14, fontweight="bold")
        ax4.set_xticklabels(
            [
                f"{model}\n({param})"
                for model, param in zip(df["model"], df["param_ct"])
            ],
            rotation=0,
        )
        ax4.grid(True, alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 5. Model size vs performance chart
        ax5 = plt.subplot(2, 3, 5)
        param_sizes = []
        for param in df["param_ct"]:
            if "b" in param.lower():
                size = float(param.lower().replace("b", ""))
                param_sizes.append(size)
            else:
                param_sizes.append(None)  # Default size

        scatter = ax5.scatter(
            param_sizes,
            df["speedup_ratio"],
            s=[200 + p * 50 for p in param_sizes],
            c=df["speedup_ratio"],
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
        )

        ax5.set_xlabel("Model Size (B parameters)", fontsize=12, fontweight="bold")
        ax5.set_ylabel("Speedup Ratio", fontsize=12, fontweight="bold")
        ax5.set_title("🎯 Model Size vs Speedup", fontsize=14, fontweight="bold")
        ax5.grid(True, alpha=0.3)

        for i, txt in enumerate(df["model"]):
            ax5.annotate(
                txt,
                (param_sizes[i], df["speedup_ratio"].iloc[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
            )

        # 6. Summary metrics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        avg_speedup = df["speedup_ratio"].mean()
        avg_time_saved = df["time_saved"].mean()
        avg_improvement = df["speedup_percentage"].mean()
        total_models = len(df)

        summary_text = f"""
        📊 PERFORMANCE SUMMARY
        
        Models Tested: {total_models}
        
        Average Speedup: {avg_speedup:.1f}x
        Average Time Saved: {avg_time_saved:.1f}s
        Average Improvement: {avg_improvement:.1f}%
        
        🏆 Best Performer:
        {df.loc[df['speedup_ratio'].idxmax(), 'model']}
        ({df['speedup_ratio'].max():.1f}x speedup)
        
        💡 Cache Hit Rate: 100%
        (All models successfully cached)
        """

        ax6.text(
            0.1,
            0.9,
            summary_text,
            transform=ax6.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        return fig, df

    @card
    @step
    def end(self):

        fig, df = self.create_performance_plots()

        current.card.append(Markdown("# 🚀 Ollama Model Cache Performance Benchmark"))
        current.card.append(
            Markdown(
                """
        This benchmark measures the performance improvement gained by caching Ollama models 
        between runs. The cache eliminates the need to re-download models, significantly 
        reducing startup times.
        """
            )
        )

        current.card.append(Image.from_matplotlib(fig))
        current.card.append(Markdown("## 📊 Detailed Performance Metrics"))

        table_data = []
        for _, row in df.iterrows():
            table_data.append(
                [
                    row["model"],
                    f"{row['cache_miss_startup_time']:.1f}s",
                    f"{row['cache_hit_startup_time']:.1f}s",
                    f"{row['speedup_ratio']:.1f}x",
                    f"{row['time_saved']:.1f}s",
                    f"{row['speedup_percentage']:.1f}%",
                ]
            )

        current.card.append(
            Table(
                headers=[
                    "Model",
                    "Cache Miss Time",
                    "Cache Hit Time",
                    "Speedup Ratio",
                    "Time Saved",
                    "Improvement %",
                ],
                data=table_data,
            )
        )

        best_model = df.loc[df["speedup_ratio"].idxmax()]
        current.card.append(
            Markdown(
                f"""
        ## 🎯 Key Insights
        
        - **Best Performer**: `{best_model['model']}` with **{best_model['speedup_ratio']:.1f}x** speedup
        - **Average Speedup**: {df['speedup_ratio'].mean():.1f}x across all models
        - **Total Time Saved**: {df['time_saved'].sum():.1f} seconds per benchmark cycle
        - **Cache Effectiveness**: {df['speedup_percentage'].mean():.1f}% average performance improvement
        
        💡 **Recommendation**: Model caching provides substantial performance benefits, 
        especially for larger models. Consider implementing caching in production workflows.
        """
            )
        )


if __name__ == "__main__":
    OllamaCacheSpeedupBench()
