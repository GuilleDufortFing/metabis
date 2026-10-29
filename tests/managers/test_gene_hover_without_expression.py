"""A comparison run without expression data leaves a gene with no statistics.

A subsystem graph draws the genes of its reactions' rules whether expression was
loaded or not. Without it the comparison has no gene results, and a gene's hover
reads that it has no comparison data, rather than an error card.
"""

from metabis.managers.data_provider import DataProvider


class _ReactionsOnly:
    def get_test_results(self, for_genes: bool = False):
        return None if for_genes else object()

    def get_statistical_info(self, element_id: str, for_genes: bool = False):
        raise ValueError("No test results found. Run the test first with run_test().")


def test_a_gene_has_no_comparison_data_when_the_comparison_tested_no_genes():
    provider = DataProvider.__new__(DataProvider)
    provider.group_comparison_artifact = _ReactionsOnly()

    assert provider.get_gene_group_comparison_data("ENSG00000130066") == {"has_comparison_data": False}
