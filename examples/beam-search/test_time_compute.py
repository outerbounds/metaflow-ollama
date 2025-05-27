import asyncio
import concurrent.futures
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from ollama import Client

# from sentence_transformers import SentenceTransformer

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

nltk.download("punkt")
nltk.download("punkt_tab")


class BeamStatus:
    """Enumeration of possible beam states."""

    RUNNING = "RUNNING"
    PRUNED = "PRUNED"
    COMPLETED = "COMPLETED"
    DUPLICATE = "DUPLICATE"
    ERROR = "ERROR"


@dataclass
class GenResult:
    """Stores the result of a single generation step."""

    next_texts: List[str]
    lookahead_texts: List[str]
    stop_reasons: List[str]
    completion_tokens: int


@dataclass
class BeamAnalysis:
    """Stores analysis of why a beam was scored/pruned as it was."""

    iteration: int
    score: float
    score_breakdown: Dict[str, float]
    pruned: bool
    pruned_reason: Optional[str] = None
    rank_at_iteration: Optional[int] = None
    current_text: Optional[str] = None


@dataclass
class Beam:
    """Represents a single beam in the search."""

    prompt: str
    index: int
    current_text: str
    score: float
    next_texts: Optional[List[str]] = None
    lookahead_texts: Optional[List[str]] = None
    stop_reasons: Optional[List[str]] = None

    # Status
    status: str = BeamStatus.RUNNING
    pruned: bool = False
    completed: bool = False

    # Additional tracking
    history: List[str] = field(default_factory=list)
    completion_tokens: int = 0
    analysis: List[BeamAnalysis] = field(default_factory=list)

    def add_analysis(self, analysis: BeamAnalysis):
        self.analysis.append(analysis)

    def get_pruning_explanation(self) -> str:
        """Get human-readable explanation of the beam's journey."""
        explanation = [f"Beam {self.index} Analysis:"]
        explanation.append(f"Final text length: {len(self.current_text.split())} words")
        explanation.append(f"Final status: {self.status}")
        for step in self.analysis:
            explanation.append(f"Iteration {step.iteration}:")
            explanation.append("Scores:")
            for metric, sc in step.score_breakdown.items():
                explanation.append(f"  - {metric}: {sc:.3f}")
            explanation.append(f"Final score: {step.score:.3f}")
            if step.rank_at_iteration is not None:
                explanation.append(f"Ranked #{step.rank_at_iteration + 1}")
            if step.pruned:
                explanation.append(f"PRUNED: {step.pruned_reason}")
        return "\n".join(explanation)


class aLaCarteRewardModel:
    """Reward model combining multiple scoring metrics."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            # 'length': 0.25,
            "coherence": 0.25,
            "diversity": 0.25,
            "relevance": 0.50,
            # 'similarity': 0.25
        }
        # try:
        # self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # logger.debug("SentenceTransformer loaded successfully.")
        # except Exception as e:
        # logger.exception("Error loading SentenceTransformer: %s", e)
        # raise e

    def _score_length(self, text: str) -> float:
        words = len(text.split())
        if words < 50:
            return words / 50.0
        elif words > 200:
            return max(0.0, 1.0 - ((words - 200) / 200.0))
        return 1.0

    def _score_coherence(self, text: str) -> float:
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return 0.5
        coherence_score = 1.0
        for sent in sentences:
            words = sent.split()
            if len(words) < 3:
                coherence_score *= 0.8
            if not sent[0].isupper():
                coherence_score *= 0.9
            if sent[-1] not in ".!?":
                coherence_score *= 0.9
        return coherence_score

    def _score_diversity(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        unique_words = len(set(words))
        return min(1.0, (unique_words / len(words)) * 2)

    def _score_relevance(self, prompt: str, completion: str) -> float:
        prompt_terms = set(re.findall(r"\w+", prompt.lower()))
        completion_terms = set(re.findall(r"\w+", completion.lower()))
        if not prompt_terms:
            return 0.5
        overlap = len(prompt_terms.intersection(completion_terms))
        return min(1.0, overlap / len(prompt_terms))

    # def _score_embedding(self, prompt: str, completion: str) -> float:
    #     try:
    #         prompt_emb = self.sentence_model.encode(prompt)
    #         completion_emb = self.sentence_model.encode(completion)
    #         norm_prompt = np.linalg.norm(prompt_emb)
    #         norm_completion = np.linalg.norm(completion_emb)
    #         if norm_prompt < 1e-6 or norm_completion < 1e-6:
    #             return 0.0
    #         similarity = np.dot(prompt_emb, completion_emb) / (norm_prompt * norm_completion)
    #         return float(similarity)
    #     except Exception as e:
    #         # logger.exception("Error computing embedding similarity: %s", e)
    #         return 0.0

    def score_with_breakdown(
        self, prompt: str, completion: str
    ) -> Tuple[float, Dict[str, float]]:
        scores = {
            # 'length': self._score_length(completion),
            "coherence": self._score_coherence(completion),
            "diversity": self._score_diversity(completion),
            "relevance": self._score_relevance(prompt, completion),
            # 'similarity': self._score_embedding(prompt, completion)
        }
        final_score = sum(
            score * self.weights.get(metric, 0) for metric, score in scores.items()
        )
        return final_score, scores


class OllamaBeamSearch:
    """
    A beam search implemnentation based on this example:
        https://github.com/huggingface/search-and-learn/blob/169fcc8a8c684f373a6d2630d28650859199a935/src/sal/search/beam_search.py
        https://arxiv.org/abs/2408.03314 ~ Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
    """

    def __init__(
        self,
        model_name: str,
        n_beams: int = 4,
        max_iterations: int = 40,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        beam_width: int = 4,
        lookahead: int = 1,
        n_threads: int = 4,
        filter_duplicates: bool = True,
        sort_completed: bool = True,
        system_prompt: Optional[str] = None,
        pruning_threshold_ratio: float = 0.85,
        early_stopping_patience: int = 5,
    ):
        self.client = Client()
        self.model_name = model_name
        self.n_beams = n_beams
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.beam_width = beam_width
        self.lookahead = lookahead
        self.filter_duplicates = filter_duplicates
        self.sort_completed = sort_completed
        self.pruning_threshold_ratio = pruning_threshold_ratio
        self.early_stopping_patience = early_stopping_patience

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_threads)
        self.reward_model = aLaCarteRewardModel()
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Keep track of global texts for duplicate filtering
        self.global_generated_texts = set()

        # logger.info("Initialized OllamaBeamSearch with model %s, n_beams=%d, max_iterations=%d",
        #             model_name, n_beams, max_iterations)

    def _get_default_system_prompt(self) -> str:
        default_prompt = (
            "You are a creative writing assistant. Follow these guidelines:\n\n"
            "1. Stay focused on the main narrative\n"
            "2. Maintain consistent characters and setting\n"
            "3. Show, don't tell\n"
            "4. Use vivid, specific details\n"
            "5. Keep the pacing natural\n\n"
            "6. Continue with grammar patterns, so if the passage ends mid-sentence, continue from where it left off\n\n"
            "7. You will be given a part of a story, and it is your job to continue generating the next chunk\n\n"
            "Respond with engaging, well-structured prose that draws the reader in."
        )
        # logger.info("Using default system prompt.")
        return default_prompt

    def _clean_text_join(self, current: str, new: str) -> str:
        new = new.strip().lstrip(".")
        if current:
            current = current.rstrip()
            if current and current[-1].isalnum() and new and new[0].isalnum():
                return f"{current} {new}"
            return f"{current}{new}"
        return new

    def _format_prompt(self, user_prompt: str, current_text: str = "") -> str:
        if not current_text:
            return f"{self.system_prompt}\n\n{user_prompt}"
        return f"{self.system_prompt}\n\n{user_prompt}\n{current_text}"

    async def _generate_step(
        self, prompt: str, current_text: str, lookahead_steps: int
    ) -> GenResult:
        try:
            formatted_prompt = self._format_prompt(prompt, current_text)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.client.generate(
                    model=self.model_name,
                    prompt=formatted_prompt,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens // self.max_iterations,
                    },
                ),
            )
            next_text = response.get("response", "").strip()
            completion_tokens = len(next_text.split())

            lookahead_text = ""
            if lookahead_steps > 0:
                # Build lookahead prompt
                joined = self._clean_text_join(current_text, next_text)
                lookahead_prompt = self._format_prompt(prompt, joined)
                lookahead_response = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.generate(
                        model=self.model_name,
                        prompt=lookahead_prompt,
                        options={
                            "temperature": self.temperature * 0.5,
                            "num_predict": self.max_tokens // self.max_iterations,
                        },
                    ),
                )
                lookahead_text = lookahead_response.get("response", "").strip()

            stop_reason = "length" if completion_tokens >= self.max_tokens else None
            return GenResult(
                next_texts=[next_text],
                lookahead_texts=[lookahead_text],
                stop_reasons=[stop_reason],
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            # logger.exception("Error during generation step: %s", e)
            return GenResult(
                next_texts=[""],
                lookahead_texts=[""],
                stop_reasons=["error"],
                completion_tokens=0,
            )

    async def search(self, prompt: str) -> List[Beam]:
        # logger.info("Starting beam search for prompt: '%s'", prompt)

        # Initialize beams
        beams = [
            Beam(prompt=prompt, index=i, current_text="", score=0.0)
            for i in range(self.n_beams)
        ]
        active_beams = beams.copy()
        completed_beams = []
        pruned_beams = []

        no_improvement_counter = {beam.index: 0 for beam in beams}
        best_scores = {beam.index: 0.0 for beam in beams}

        iteration = 0
        while iteration < self.max_iterations:
            # logger.info("Iteration %d: Active beams count: %d", iteration, len(active_beams))
            if not active_beams:
                # logger.warning("No active beams remaining; terminating search loop.")
                break

            ## Generate steps
            tasks = [
                self._generate_step(
                    beam.prompt,
                    beam.current_text,
                    self.lookahead if iteration < self.max_iterations - 1 else 0,
                )
                for beam in active_beams
            ]
            gen_results = await asyncio.gather(*tasks)

            ## Log results
            new_active_beams = []
            for beam, gen_result in zip(active_beams, gen_results):
                if gen_result.next_texts and gen_result.next_texts[0]:
                    # Join text
                    beam.current_text = self._clean_text_join(
                        beam.current_text, gen_result.next_texts[0]
                    )
                    beam.history.append(gen_result.next_texts[0])
                    beam.completion_tokens += gen_result.completion_tokens

                    # Check duplicates
                    if self.filter_duplicates:
                        if beam.current_text in self.global_generated_texts:
                            beam.pruned = True
                            beam.status = BeamStatus.DUPLICATE
                            pruned_beams.append(beam)
                            # logger.info("Beam %d pruned due to duplicate text.", beam.index)
                            continue
                        else:
                            self.global_generated_texts.add(beam.current_text)

                # If we hit a stop reason or got empty text, mark completed
                if (
                    gen_result.stop_reasons and gen_result.stop_reasons[0] == "length"
                ) or not gen_result.next_texts[0]:
                    beam.completed = True
                    beam.status = BeamStatus.COMPLETED
                    completed_beams.append(beam)
                    # logger.info("Beam %d marked as completed.", beam.index)
                else:
                    # Keep beam active if not pruned
                    if not beam.pruned:
                        new_active_beams.append(beam)

            ## Score each active beam
            for beam in new_active_beams:
                score, breakdown = self.reward_model.score_with_breakdown(
                    beam.prompt, beam.current_text
                )
                beam.score = score

                if score - best_scores[beam.index] < 1e-3:
                    no_improvement_counter[beam.index] += 1
                else:
                    no_improvement_counter[beam.index] = 0
                    best_scores[beam.index] = score

                analysis = BeamAnalysis(
                    iteration=iteration,
                    score=score,
                    score_breakdown=breakdown,
                    pruned=False,
                    rank_at_iteration=None,
                    current_text=beam.current_text,
                )
                beam.add_analysis(analysis)

            ## Early stopping
            still_active = []
            for beam in new_active_beams:
                if no_improvement_counter[beam.index] >= self.early_stopping_patience:
                    beam.pruned = True
                    beam.status = BeamStatus.PRUNED
                    pruned_beams.append(beam)
                    # logger.info("Beam %d pruned due to early stopping.", beam.index)
                else:
                    still_active.append(beam)
            new_active_beams = still_active

            ## Threshold-based pruning
            if new_active_beams:
                best_score_in_iteration = max(b.score for b in new_active_beams)
                threshold = best_score_in_iteration * self.pruning_threshold_ratio
                kept = []
                for beam in new_active_beams:
                    if beam.score < threshold:
                        beam.pruned = True
                        beam.status = BeamStatus.PRUNED
                        reason = (
                            f"score {beam.score:.3f} below threshold {threshold:.3f}"
                        )
                        if beam.analysis:
                            beam.analysis[-1].pruned = True
                            beam.analysis[-1].pruned_reason = reason
                        pruned_beams.append(beam)
                        # logger.info("Beam %d pruned: %s.", beam.index, reason)
                    else:
                        kept.append(beam)
                new_active_beams = kept

            active_beams = new_active_beams
            iteration += 1

        # Merge completed, active, AND pruned beams
        all_beams = completed_beams + active_beams + pruned_beams

        if not all_beams:
            # logger.error("No beams produced a valid completion.")
            return [
                Beam(
                    prompt=prompt,
                    index=0,
                    current_text="No valid completions generated.",
                    score=0.0,
                )
            ]

        # Sort all beams by final score
        all_beams.sort(key=lambda b: b.score, reverse=True)

        # If desired, slice to top n_beams
        if self.sort_completed:
            # logger.info("Sorting final beams by score.")
            pass
        if len(all_beams) > self.n_beams:
            all_beams = all_beams[: self.n_beams]

        # Add final analysis for each selected beam
        for rank, beam in enumerate(all_beams):
            final_score, final_breakdown = self.reward_model.score_with_breakdown(
                beam.prompt, beam.current_text
            )
            final_analysis = BeamAnalysis(
                iteration=self.max_iterations,
                score=final_score,
                score_breakdown=final_breakdown,
                pruned=beam.pruned,
                rank_at_iteration=rank,
                current_text=beam.current_text,
            )
            beam.add_analysis(final_analysis)

        # logger.info("Beam search completed. Returning %d beams.", len(all_beams))
        # Return all beams (including pruned or duplicates) so user can see them
        return all_beams


def run_search(prompt: str, beam_search: OllamaBeamSearch) -> List[Beam]:
    """Run the beam search synchronously for a given prompt."""
    # logger.info("Running beam search for prompt: '%s'", prompt)
    return asyncio.run(beam_search.search(prompt))


# Example usage:
# if __name__ == "__main__":
#     beam_search = OllamaBeamSearch(
#         model_name="llama3.2",
#         n_beams=6,
#         max_iterations=3,
#         pruning_threshold_ratio=0.9
#     )
#     results = run_search("Write a story about an artificial specific intelligence.", beam_search)
#     for i, beam in enumerate(results, start=1):
#         print(f"\n=== Beam {i} ===")
#         print("Text:", beam.current_text)
#         print("Score:", beam.score)
#         print("Status:", beam.status)
#         print(beam.get_pruning_explanation())
