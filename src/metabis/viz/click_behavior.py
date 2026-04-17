"""Shared click behavior helpers for Bokeh plots."""

from typing import Any, Dict

from bokeh.models import CustomJS, Plot, TapTool

CLICK_BEHAVIOR_TAG = "metabis_click_behavior_attached"


CLICK_BEHAVIOR_CODE = r"""
const selection = cb_obj && Array.isArray(cb_obj.indices) ? cb_obj : source.selected;
const indices = selection && Array.isArray(selection.indices)
    ? selection.indices
    : source.selected.indices;
if (!indices.length) {
    return;
}

const idx = indices[0];
const data = source.data || {};

const pickFirst = (fields) => {
    for (const field of fields) {
        if (!(field in data)) {
            continue;
        }

        const column = data[field];
        if (!column || idx >= column.length) {
            continue;
        }

        const value = column[idx];
        if (value === null || value === undefined || value === "") {
            continue;
        }

        return value;
    }

    return null;
};

const clearSelection = () => {
    setTimeout(() => {
        const activeSelection = cb_obj && Array.isArray(cb_obj.indices)
            ? cb_obj
            : source.selected;
        activeSelection.indices = [];
    }, 0);
};

try {
    const elementId = String(
        pickFirst([
            "element_id",
            "id",
            "reaction_id",
            "gene_id",
            "feature_name",
            "reaction",
            "gene",
        ]) || `element_${idx}`
    );
    const elementLabel = String(
        pickFirst([
            "display_label",
            "label",
            "reaction_id",
            "gene_id",
            "feature_name",
            "reaction",
            "gene",
            "subsystem",
        ]) || elementId
    );
    const hoverContent = pickFirst(["hover", "desc", "hover_text"]);

    const downloadEnabled = (
        typeof enabled_toggle !== "undefined" && enabled_toggle && enabled_toggle.active
    );

    const sanitizeName = (value) => {
        const cleaned = String(value || "")
            .replace(/[^a-z0-9]/gi, "_")
            .replace(/^_+|_+$/g, "")
            .substring(0, 50);
        return cleaned || "element";
    };

    const downloadHoverPanel = () => {
        if (!hoverContent) {
            return false;
        }

        const safeName = sanitizeName(elementLabel || elementId);
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${elementLabel} - Details</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .header {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            color: #2c3e50;
            font-size: 24px;
        }
        .header .subtitle {
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 5px;
        }
        .content {
            line-height: 1.6;
        }
        .content > div {
            max-width: 100% !important;
        }
        .footer {
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            font-size: 11px;
            color: #95a5a6;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>${elementLabel}</h1>
            <div class="subtitle">ID: ${elementId}</div>
        </div>
        <div class="content">
            ${hoverContent}
        </div>
        <div class="footer">
            Exported from Metabis on ${new Date().toLocaleString()}
        </div>
    </div>
</body>
</html>`;

        const blob = new Blob([html], { type: "text/html;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${prefix}_${safeName}.html`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        return true;
    };

    const navigateToSubsystem = () => {
        if (typeof subsystem_select === "undefined" || !subsystem_select) {
            return false;
        }

        const subsystemFlags = data.is_subsystem;
        const isSubsystem = Boolean(
            subsystemFlags && idx < subsystemFlags.length && subsystemFlags[idx]
        );
        if (!isSubsystem) {
            return false;
        }

        let subsystemName = elementLabel.replace(/\n/g, " ").replace(/\s+/g, " ").trim();
        if (!subsystemName) {
            return false;
        }

        const isTruncated = subsystemName.endsWith("...");
        const searchPrefix = isTruncated ? subsystemName.slice(0, -3).trim() : null;
        const options = subsystem_select.options || [];
        let matchedValue = null;

        for (const option of options) {
            const optionValue = Array.isArray(option) ? option[0] : option;
            const optionLabel = Array.isArray(option) ? option[1] : option;
            const normalizedLabel = String(optionLabel)
                .replace(/\n/g, " ")
                .replace(/\s+/g, " ")
                .trim();
            const normalizedValue = String(optionValue)
                .replace(/\n/g, " ")
                .replace(/\s+/g, " ")
                .trim();

            if (normalizedLabel === subsystemName || normalizedValue === subsystemName) {
                matchedValue = optionValue;
                break;
            }

            if (
                isTruncated &&
                searchPrefix &&
                (normalizedLabel.startsWith(searchPrefix) || normalizedValue.startsWith(searchPrefix))
            ) {
                matchedValue = optionValue;
                break;
            }
        }

        if (matchedValue === null) {
            return false;
        }

        subsystem_select.value = matchedValue;
        subsystem_select.change.emit();
        return true;
    };

    const openExternalLink = () => {
        const externalUrl = pickFirst(["external_url"]);
        if (!externalUrl) {
            return false;
        }

        const link = document.createElement("a");
        link.href = String(externalUrl);
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        return true;
    };

    if (downloadEnabled) {
        downloadHoverPanel();
        clearSelection();
        return;
    }

    if (navigateToSubsystem()) {
        clearSelection();
        return;
    }

    if (openExternalLink()) {
        clearSelection();
        return;
    }

    clearSelection();
} catch (error) {
    console.error("Metabis click behavior failed", error, { data, indices });
    clearSelection();
}
"""


def get_external_reference_data(
    data_manager: Any,
    element_id: Any,
    element_type: str,
) -> Dict[str, str]:
    """Get display and external reference data for a reaction or gene."""
    element_id_str = str(element_id)
    metabolic_model = getattr(data_manager, "metabolic_model", None)
    if metabolic_model is None:
        return {
            "element_id": element_id_str,
            "display_label": element_id_str,
            "external_url": "",
        }

    if element_type == "gene":
        reference_data = metabolic_model.get_gene_reference_data(element_id_str)
    elif element_type == "reaction":
        reference_data = metabolic_model.get_reaction_reference_data(element_id_str)
    else:
        reference_data = {
            "raw_id": element_id_str,
            "display_label": element_id_str,
            "external_url": "",
        }

    return {
        "element_id": reference_data.get("raw_id", element_id_str),
        "display_label": reference_data.get("display_label", element_id_str),
        "external_url": reference_data.get("external_url", ""),
    }


def add_click_behavior(
    model: Any,
    filename_prefix: str = "element",
    enabled_toggle: Any = None,
    subsystem_select: Any = None,
) -> Any:
    """Attach click behavior to eligible Bokeh data sources in a model tree."""
    references = {model}
    if hasattr(model, "references"):
        references.update(model.references())

    for plot in references:
        if not isinstance(plot, Plot):
            continue

        eligible_sources = []
        seen_sources = set()
        for renderer in getattr(plot, "renderers", []):
            source = getattr(renderer, "data_source", None)
            if source is None:
                continue

            source_id = id(source)
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)

            data = getattr(source, "data", {}) or {}
            has_hover = any(field in data for field in ("hover", "desc", "hover_text"))
            has_links = any(bool(value) for value in data.get("external_url", []))
            has_subsystem_navigation = (
                subsystem_select is not None and "is_subsystem" in data
            )

            if (
                has_links
                or has_subsystem_navigation
                or (enabled_toggle is not None and has_hover)
            ):
                eligible_sources.append(source)

        if not eligible_sources:
            continue

        if not plot.select(type=TapTool):
            plot.add_tools(TapTool())

        for source in eligible_sources:
            existing_tags = list(getattr(source, "tags", []) or [])
            if CLICK_BEHAVIOR_TAG in existing_tags:
                continue

            source.tags = existing_tags + [CLICK_BEHAVIOR_TAG]

            callback_args = {"source": source, "prefix": filename_prefix}
            if enabled_toggle is not None:
                callback_args["enabled_toggle"] = enabled_toggle
            if subsystem_select is not None:
                callback_args["subsystem_select"] = subsystem_select

            source.selected.js_on_change(
                "indices",
                CustomJS(args=callback_args, code=CLICK_BEHAVIOR_CODE),
            )

    return model
