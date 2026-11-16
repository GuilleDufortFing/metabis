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
from metabis.domain.reaction_data import ReactionData, _identifier_spellings

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
# the contests present in the bundled Th17 data.
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


# --- Identifiers Human1 lists, and Recon2's spelling of them ----------------


def recon2(tmp_path, model, columns, name="recon2.tsv"):
    """Load a dual Recon2-named file read through Human1's Recon3D annotation."""
    return load(
        write_dual(tmp_path, columns, name=name),
        model,
        reaction_name_format="rxnRecon3DID",
        check_old_identifiers=True,
    )


def test_a_recon2_compartment_is_also_read_in_recon3d_brackets():
    assert _identifier_spellings("EX_glc(e)") == ["EX_glc(e)", "EX_glc[e]"]
    assert _identifier_spellings("GLCt1r") == ["GLCt1r"]


def test_an_identifier_listed_beside_others_maps_to_its_reaction(tmp_path, model):
    # Human1 annotates MAR08572 as "HMR_8572;RE0944C". No file carries that
    # string; a Recon2 file carries RE0944C.
    frame = recon2(tmp_path, model, {"RE0944C_pos": [1.0, 2.0, 3.0]}).get_reaction_df()

    assert list(frame.columns) == ["MAR08572"]


def test_a_recon2_exchange_maps_through_its_recon3d_spelling(tmp_path, model):
    # Recon2 writes EX_10fthf(e); Human1's Recon3D annotation, EX_10fthf[e].
    frame = recon2(
        tmp_path, model, {"EX_10fthf(e)_pos": [1.0, 2.0, 3.0]}
    ).get_reaction_df()

    assert list(frame.columns) == ["MAR09205"]


def test_a_whole_annotation_keeps_its_reaction_over_a_list_naming_another(
    tmp_path, model
):
    # RE3422C is MAR01334's whole annotation and one entry of MAR01338's
    # "HMR_1338;RE3422C". The whole match decides, as it did before lists were read.
    frame = recon2(tmp_path, model, {"RE3422C_pos": [1.0, 2.0, 3.0]}).get_reaction_df()

    assert list(frame.columns) == ["MAR01334"]


def test_human1s_own_identifier_outranks_the_legacy_table(tmp_path, model):
    # The legacy table lists EX_ac(e), acetate exchange, under MAR09086 and also
    # under MAR09239, the exchange of heptaglutamyl-folate, and its last claim
    # used to win. Human1's Recon3D annotation names only MAR09086.
    loaded = recon2(tmp_path, model, {"EX_ac(e)_pos": [1.0, 2.0, 3.0]})

    assert list(loaded.get_reaction_df().columns) == ["MAR09086"]
    assert loaded.mapping_stats["unmapped_count"] == 0


def test_the_legacy_table_is_not_read_under_another_spelling(tmp_path, model):
    # Respelled, EX_Rtotal(e) would meet the legacy table's EX_Rtotal[e], which
    # it files under the exchange of ribose.
    loaded = recon2(tmp_path, model, {"EX_Rtotal(e)_pos": [1.0, 2.0, 3.0]})

    assert loaded.get_reaction_df().empty
    assert loaded.mapping_stats["unmapped_count"] == 1


def test_an_sbml_identifier_is_also_read_unescaped():
    # Compass keeps Recon 2.2's SBML IDs: EX_glc(e) reaches a file escaped.
    assert _identifier_spellings("R_EX_glc_LPAREN_e_RPAREN_") == [
        "R_EX_glc_LPAREN_e_RPAREN_",
        "EX_glc(e)",
        "EX_glc[e]",
    ]


# --- The declared source model's table --------------------------------------


def declared(tmp_path, model, columns, table, name="declared.tsv", **overrides):
    return load(
        write_dual(tmp_path, columns, name=name),
        model,
        **{
            "reaction_name_format": "rxnRecon3DID",
            "check_old_identifiers": True,
            "source_reactions": table,
            **overrides,
        },
    )


def test_the_table_maps_the_ids_it_lists_and_names_their_reversals(tmp_path, model):
    loaded = declared(
        tmp_path, model, {"G3PD1_pos": [1.0, 2.0, 3.0]}, {"G3PD1": ("MAR00479", -1)}
    )

    assert list(loaded.get_reaction_df().columns) == ["MAR00479"]
    assert loaded.reaction_mapping == {"G3PD1": "MAR00479"}
    assert loaded.reversed_reactions == {"MAR00479"}
    assert loaded.mapping_stats["reversed_count"] == 1


def test_each_source_reaction_carries_its_own_orientation(tmp_path, model):
    # MAR03146's thiolase step is r0634 written backwards in Recon2, and part of the
    # lumped FAOXC101C8m written forwards: whichever backs it decides.
    table = {"r0634": ("MAR03146", -1), "FAOXC101C8m": ("MAR03146", 1)}

    specific = declared(tmp_path, model, {"r0634_pos": [1.0, 2.0, 3.0]}, table, name="a.tsv")
    lumped = declared(tmp_path, model, {"FAOXC101C8m_pos": [1.0, 2.0, 3.0]}, table, name="b.tsv")

    assert specific.reversed_reactions == {"MAR03146"}
    assert lumped.reversed_reactions == frozenset()


def test_a_table_pairing_still_counts_how_lumped_its_source_is(tmp_path, model):
    # Both back MAR03146 by the table. Human1's Recon3D annotation lists the
    # thiolase step r0634 under that reaction alone and the lumped FAOXC101C8m
    # under three, so the step backs it, although it sorts last.
    table = {"r0634": ("MAR03146", -1), "FAOXC101C8m": ("MAR03146", 1)}
    loaded = declared(
        tmp_path, model, {"FAOXC101C8m_pos": [1.0, 2.0, 3.0], "r0634_pos": [4.0, 5.0, 6.0]}, table
    )

    (record,) = loaded.mapping_stats["id_collisions"]
    assert record["kept"] == "r0634"
    assert record["discarded"] == [("FAOXC101C8m", 3)]


def test_a_table_row_with_no_human1_reaction_leaves_the_id_unmapped(tmp_path, model):
    # Recon2's r1446, butyryl-CoA dehydrogenation, under dodecenoyl-CoA hydration.
    loaded = declared(tmp_path, model, {"r1446_pos": [1.0, 2.0, 3.0]}, {"r1446": ("", 0)})

    assert loaded.get_reaction_df().empty
    assert loaded.mapping_stats["unmapped_count"] == 1


def test_an_id_the_table_does_not_list_takes_its_reactions_agreed_orientation(tmp_path, model):
    agreed = declared(
        tmp_path, model, {"G3PD1_pos": [1.0, 2.0, 3.0]}, {"OTHER": ("MAR00479", -1)}, name="a.tsv"
    )
    disputed = declared(
        tmp_path,
        model,
        {"G3PD1_pos": [1.0, 2.0, 3.0]},
        {"OTHER": ("MAR00479", -1), "ANOTHER": ("MAR00479", 1)},
        name="b.tsv",
    )

    assert agreed.reversed_reactions == {"MAR00479"}
    assert agreed.mapping_stats["unread_orientation_count"] == 0
    assert disputed.reversed_reactions == frozenset()
    assert disputed.mapping_stats["unread_orientation_count"] == 1


def test_a_single_column_reaction_to_be_reversed_keeps_its_backward_direction(tmp_path, model):
    # MAR00479 is irreversible in Human1; reversed, it is made reversible, so a
    # single-column file keeps a backward column for it as it did before.
    path = str(tmp_path / "single.tsv")
    pd.DataFrame({"G3PD1": [1.0, 2.0, 3.0]}, index=["s1", "s2", "s3"]).to_csv(
        path, sep="\t", index_label="sample_id"
    )
    options = dict(data_mode="single", reaction_name_format="rxnRecon3DID")

    reversed_ = load(path, model, **options, source_reactions={"G3PD1": ("MAR00479", -1)})
    plain = load(path, model, **options)

    assert "MAR00479_back" in reversed_.get_reaction_df().columns
    assert "MAR00479_back" not in plain.get_reaction_df().columns
