
class HHEMMetrics:
    def __init__(self):
        self.factual_consistancy_rate = None
        pass

    def compute_hallucination_rate(
            self, hhem_scores: list[float], threshold=0.5
        ):
        """
        
        """
        fcr = self.compute_factual_consistancy_rate(
            hhem_scores, threshold=threshold
        )
        hallucination_rate = 1.0 - fcr
        return hallucination_rate

    def compute_factual_consistancy_rate(
            self, hhem_scores: list[float], threshold=0.5
        ):
        """
        
        """
        factual_count = sum(score >= threshold for score in hhem_scores)
        factual_consistancy_rate = factual_count/len(hhem_scores)
        return factual_consistancy_rate

    def compute_answer_rate(self, summaries: list[str]):
        """
        
        """
        valid_summ_count = sum(
            self.is_valid_summary(summary) for summary in summaries
        )
        answer_rate = valid_summ_count/len(summaries)
        return answer_rate

    def compute_avg_summary_length(self, summaries: list[str]):
        """
        
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

        """
        if len(summary.split()) >= 5:
            return True
        else:
            return False