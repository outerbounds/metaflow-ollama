from metaflow import (
    FlowSpec,
    step,
    ollama,
    Config,
    config_expr,
    pypi,
    current,
    card,
    resources,
    environment,
)
from metaflow.cards import VegaChart


class OllamaBeamSearchFlow(FlowSpec):

    config = Config("config", default="config.json")

    @card
    @resources(**config_expr("config.resources"))
    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @pypi(packages=config_expr("config.packages"))
    @ollama(models=[config_expr("config.model")], debug=True)
    @step
    def start(self):
        import asyncio
        from test_time_compute import OllamaBeamSearch
        from viz import beams_to_dataframe, trim_pruned_rows, plot_beam_scores_altair

        self.parameters = dict(
            model_name=self.config.model,
            n_beams=self.config.n_beams,
            max_iterations=self.config.max_iterations,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        beam_search = OllamaBeamSearch(**self.parameters)
        beams = asyncio.run(beam_search.search(self.config.prompt))
        df = beams_to_dataframe(beams)
        chart = plot_beam_scores_altair(trim_pruned_rows(df))
        current.card.append(VegaChart.from_altair_chart(chart))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    OllamaBeamSearchFlow()
