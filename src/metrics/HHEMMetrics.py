from src.LLMs.AbstractLLM import (
    MODEL_RETURNED_NON_STRING_TYPE_OUTPUT, MODEL_FAILED_TO_RETURN_OUTPUT
)

class HHEMMetrics:
    """
    Specialized Metrics for HHEM output

    Attributes:
        None

    Methods:
        compute_hallucination_rate(hhem_scores, summaries, threshold)
        compute_factual_consistancry_rate(hhem_scores, summaries, threshold)
        compute_answer_rate(summaries)
        computer_avg_summary_length(summaries)
        is_valid_summary(summary)
        has_error_output(summary)
    
    """

    def __init__(self):
        pass

    def compute_hallucination_rate(
            self, hhem_scores: list[float], summaries: list[float], threshold=0.5
        ):
        """
        Computes hallucination rate with default threshold of 0.5

        Args:
            hhem_scores (list[float]): hhem scores aligned with summaries
            summaries (list[str]): summaries aligned with hhem scores
            threshold (float): confidence threshold for positive result

        Returns:
            float: hallucination rate
        """

        fcr = self.compute_factual_consistancy_rate(
            hhem_scores, summaries, threshold=threshold
        )
        hallucination_rate = 1.0 - fcr
        return hallucination_rate

    def compute_factual_consistancy_rate(
            self, hhem_scores: list[float], summaries: list[str], threshold=0.5
        ):
        """
        Computes factual consistancy rate with default threshold of 0.5

        Args:
            hhem_scores (list[float]): hhem scores aligned with summaries
            summaries (list[str]): summaries aligned with hhem scores
            threshold (float): confidence threshold for positive result

        Returns:
            float: factual consistancy rate
        
        """

        total_count = 0
        factual_count = 0
        for score, summary in zip(hhem_scores, summaries):
            if self.is_valid_summary(summary):
                total_count += 1
                if score >= threshold:
                    factual_count += 1
        factual_consistancy_rate = factual_count/total_count
        return factual_consistancy_rate

    def compute_answer_rate(self, summaries: list[str]):
        """
        Computes the the rate valid summaries. A valid summary is a summary of
        reasonable length that attempts to summarize an article.

        Args:
            summaries (list[str]): list of summaries

        Returns:
            float: answer rate
        """

        valid_summ_count = sum(
            self.is_valid_summary(summary) for summary in summaries
        )
        answer_rate = valid_summ_count/len(summaries)
        return answer_rate

    def compute_avg_summary_length(self, summaries: list[str]):
        """
        Computes average summary length for all articles

        Args:
            summaries (list[str]): list of summaries

        Returns:
            float: Average summary length
        """

        summary_lengths = []
        for summary in summaries:
            if self.is_valid_summary(summary):
                summary_length = len(summary.split())
                summary_lengths.append(summary_length)
        avg_summary_length = sum(summary_lengths)/len(summary_lengths)
        return avg_summary_length

    def is_valid_summary(self, summary: str):
        """
        Checks if summary is valid and returns True if it is else False

        Args:
            summary (str): the summary

        Returns:
            bool: True if valid summary else False


        """

        if self.has_error_output(summary):
            return False
        elif len(summary.split()) >= 5:
            return True
        else:
            return False

    def has_error_output(self, summary: str):
        """
        Detects if summary contains error output and returns True if so

        Args:
            summary (str): the summary

        Returns:
            bool: True if summary is exact error output string
        """

        if (
            summary == MODEL_FAILED_TO_RETURN_OUTPUT or
            summary == MODEL_RETURNED_NON_STRING_TYPE_OUTPUT
        ):
            return True
        else:
            return False