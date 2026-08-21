"""Which source reaction backs a Human1 reaction, and which directions survive.

WP10 covers two silent data-loss defects in the dual-mode loader. D-66: the
crosswalk is many-to-many, so several source reactions can claim one Human1
reaction, and the loop that installs their values let the last one in file order
win with no report. D-67: the loader had no ``elif neg_exists`` branch, so a
source reaction measured only in reverse was discarded entirely.
"""

import pandas as pd
import pytest

from metabis import MetabolicModel, ReactionPreprocessingConfig, ReactionTransform
from metabis.domain.reaction_data import ReactionData

resolve = ReactionData._resolve_reaction_id_collisions


def claim(source_id, mapping_degree, high_effort=False):
    return {
        "source_id": source_id,
        "mapping_degree": mapping_degree,
        "high_effort": high_effort,
    }


@pytest.fixture(scope="module")
def model() -> MetabolicModel:
    return MetabolicModel()


def write_dual(tmp_path, columns, name="reactions.tsv"):
    """Write a dual-mode reaction file: samples in rows, one row per sample."""
    frame = pd.DataFrame(columns, index=["s1", "s2", "s3"])
    path = tmp_path / name
    frame.to_csv(path, sep="\t", index_label="sample_id")
    return str(path)


def load(path, model, **overrides):
    options = dict(
        data_mode="dual",
        metabolic_model=model,
        reaction_name_format="mar_id",
        csv_separator="\t",
        samples_in="rows",
        preprocessing=ReactionPreprocessingConfig(transform=ReactionTransform.NONE),
    )
    options.update(overrides)
    return ReactionData(path, **options)


# --- D-67 -------------------------------------------------------------------


def test_all_three_directional_shapes_survive_dual_loading(tmp_path, model):
    path = write_dual(
        tmp_path,
        {
            "MAR03905_pos": [1.0, 2.0, 3.0],
            "MAR03905_neg": [4.0, 5.0, 6.0],
            "MAR03907_pos": [7.0, 8.0, 9.0],
            # Measured only in reverse: Compass could not make the forward
            # direction carry flux, or the run simply reported one side.
            "MAR04281_neg": [10.0, 11.0, 12.0],
        },
    )

    frame = load(path, model).get_reaction_df()

    assert "MAR03905" in frame.columns and "MAR03905_back" in frame.columns
    assert "MAR03907" in frame.columns and "MAR03907_back" not in frame.columns
    # The shape that used to vanish: no forward column at all.
    assert "MAR04281_back" in frame.columns
    assert "MAR04281" not in frame.columns
    assert frame["MAR04281_back"].tolist() == [10.0, 11.0, 12.0]


def test_a_reverse_only_reaction_keeps_the_magnitude_of_its_penalty(tmp_path, model):
    # `_neg` columns are stored signed by some producers; the loader takes the
    # magnitude, exactly as it does for the two-sided case.
    path = write_dual(tmp_path, {"MAR04281_neg": [-3.0, -4.0, -5.0]})

    frame = load(path, model).get_reaction_df()

    assert frame["MAR04281_back"].tolist() == [3.0, 4.0, 5.0]


# --- D-66 -------------------------------------------------------------------


def test_the_most_specific_source_reaction_represents_a_contested_reaction():
    # The real case from the bundled Th17 data: FAOXC80 is a lumped Recon2
    # reaction listed as a synonym of twelve Human1 reactions, so it is a poor
    # representative of any single one. FAOXC61C4m claims three and wins.
    candidates = {
        "MAR03160": [
            claim("FAOXC80", 12, high_effort=True),
            claim("FAOXC6C4m", 4, high_effort=True),
            claim("FAOXC61C4m", 3, high_effort=True),
        ]
    }

    mapping, collisions = resolve(candidates)

    assert mapping == {"FAOXC61C4m": "MAR03160"}
    assert collisions == [
        {
            "mar_id": "MAR03160",
            "kept": "FAOXC61C4m",
            "kept_degree": 3,
            "discarded": [("FAOXC6C4m", 4), ("FAOXC80", 12)],
        }
    ]


def test_an_exact_identifier_match_outranks_a_legacy_one_at_equal_specificity():
    candidates = {
        "MAR00444": [
            claim("GLYCTDle", 1, high_effort=True),
            claim("GLYCt", 1, high_effort=False),
        ]
    }

    mapping, _ = resolve(candidates)

    assert mapping == {"GLYCt": "MAR00444"}


def test_a_remaining_tie_breaks_lexicographically_not_by_file_order():
    forwards = {"MAR00001": [claim("alpha", 2), claim("beta", 2)]}
    backwards = {"MAR00001": [claim("beta", 2), claim("alpha", 2)]}

    assert resolve(forwards)[0] == resolve(backwards)[0] == {"alpha": "MAR00001"}


def test_an_uncontested_reaction_produces_no_collision_record():
    mapping, collisions = resolve({"MAR00001": [claim("only", 7)]})

    assert mapping == {"only": "MAR00001"}
    assert collisions == []


# `GLYCTDle` and `GLYCt` both claim `MAR00444` in the shipped crosswalk - one of
# the 39 contests present in the bundled Th17 data.
CONTESTED = {
    "GLYCTDle_pos": [1.0, 1.0, 1.0],
    "GLYCt_pos": [9.0, 9.0, 9.0],
}


def test_column_order_cannot_change_which_source_backs_a_reaction(tmp_path, model):
    # Written in both orders. Before A-19 the values installed came from
    # whichever column appeared last in the file.
    forward = load(
        write_dual(tmp_path, CONTESTED, name="forward.tsv"),
        model,
        reaction_name_format="rxnBiGGID",
        check_old_identifiers=True,
    )
    reversed_order = load(
        write_dual(tmp_path, dict(reversed(CONTESTED.items())), name="reversed.tsv"),
        model,
        reaction_name_format="rxnBiGGID",
        check_old_identifiers=True,
    )

    forward_frame = forward.get_reaction_df()
    assert "MAR00444" in forward_frame.columns
    pd.testing.assert_series_equal(
        forward_frame["MAR00444"],
        reversed_order.get_reaction_df()["MAR00444"],
        check_names=False,
    )
    assert (
        forward.mapping_stats["id_collisions"]
        == reversed_order.mapping_stats["id_collisions"]
    )


def test_the_load_report_counts_contested_reactions_and_their_sources(tmp_path, model):
    contested = load(
        write_dual(tmp_path, CONTESTED, name="contested.tsv"),
        model,
        reaction_name_format="rxnBiGGID",
        check_old_identifiers=True,
    )

    assert contested.mapping_stats["contested_reaction_count"] == 1
    assert contested.mapping_stats["contesting_source_count"] == 2
    assert contested.mapping_stats["id_collisions"] == [
        {
            "mar_id": "MAR00444",
            "kept": "GLYCt",
            "kept_degree": 1,
            "discarded": [("GLYCTDle", 1)],
        }
    ]

    uncontested = load(
        write_dual(tmp_path, {"MAR03905_pos": [1.0, 1.0, 1.0]}, name="clean.tsv"), model
    )
    assert uncontested.mapping_stats["contested_reaction_count"] == 0
    assert uncontested.mapping_stats["contesting_source_count"] == 0
    assert uncontested.mapping_stats["id_collisions"] == []


def test_mapping_is_injective_so_no_source_can_overwrite_another(tmp_path, model):
    reaction_data = load(
        write_dual(tmp_path, CONTESTED, name="injective.tsv"),
        model,
        reaction_name_format="rxnBiGGID",
        check_old_identifiers=True,
    )
    mapping, _, _, _ = reaction_data._map_base_reactions_to_model(
        list(CONTESTED), "rxnBiGGID", "dual", True
    )

    assert len(set(mapping.values())) == len(mapping)
    # And only the winner's values reach the frame.
    assert reaction_data.get_reaction_df()["MAR00444"].tolist() == [9.0, 9.0, 9.0]
