"""A hover violin's group names are text in its SVG, whatever characters they hold."""

from metabis.viz.plot_violins import create_static_violin_for_hover


def test_group_names_are_escaped_in_the_violins_svg():
    svg = create_static_violin_for_hover(
        [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], "#000000", "#ffffff", group1_name="CD4 & CD8", group2_name="<b>Th17</b>",
        size=(300, 220),
    )

    assert "CD4 &amp; CD8" in svg and "&lt;b&gt;Th17&lt;/b&gt;" in svg
    assert "<b>" not in svg
