"""Helpers for rendering linked labels and numbers in Bokeh data tables."""

import math
from typing import Any

from bokeh.models import HTMLTemplateFormatter


def format_statistical_value(value: Any) -> str:
    """Format a numeric statistical value with compact, readable precision."""
    if value is None:
        return ""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(numeric_value):
        return ""
    if not math.isfinite(numeric_value):
        return str(numeric_value)
    if numeric_value == 0:
        return "0"

    abs_value = abs(numeric_value)
    if abs_value < 0.01:
        mantissa, exponent = f"{numeric_value:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"

    exponent = math.floor(math.log10(abs_value))
    decimal_places = 2 - exponent
    if decimal_places >= 0:
        rounded_value = round(numeric_value, decimal_places)
        if rounded_value == 0:
            return "0"
        formatted = f"{rounded_value:.{decimal_places}f}"
        return formatted.rstrip("0").rstrip(".")

    rounding_factor = 10 ** (-decimal_places)
    rounded_value = round(numeric_value / rounding_factor) * rounding_factor
    if rounded_value == 0:
        return "0"
    return f"{rounded_value:.0f}"


def build_statistical_number_formatter() -> HTMLTemplateFormatter:
    """Build a formatter that mirrors the statistical value display rules."""
    return HTMLTemplateFormatter(
        template=r"""
        <%
        function formatStatisticalValue(rawValue) {
            if (rawValue === null || rawValue === undefined) {
                return "";
            }

            let numericValue = rawValue;
            if (typeof numericValue !== "number") {
                numericValue = Number(numericValue);
                if (Number.isNaN(numericValue)) {
                    return String(rawValue);
                }
            }

            if (!Number.isFinite(numericValue)) {
                return String(numericValue);
            }
            if (numericValue === 0) {
                return "0";
            }

            const absValue = Math.abs(numericValue);
            if (absValue < 0.01) {
                let scientific = numericValue.toExponential(2);
                scientific = scientific.replace(/\.?0+e/, "e");
                scientific = scientific.replace(/e\+?(-?)0*(\d+)/, "e$1$2");
                return scientific;
            }

            const exponent = Math.floor(Math.log10(absValue));
            const decimalPlaces = 2 - exponent;

            if (decimalPlaces >= 0) {
                const roundedValue = Number(numericValue.toFixed(decimalPlaces));
                if (roundedValue === 0) {
                    return "0";
                }
                return roundedValue
                    .toFixed(decimalPlaces)
                    .replace(/\.0+$/, "")
                    .replace(/(\.\d*?)0+$/, "$1")
                    .replace(/\.$/, "");
            }

            const roundingFactor = Math.pow(10, -decimalPlaces);
            const roundedValue = Math.round(numericValue / roundingFactor) * roundingFactor;
            if (roundedValue === 0) {
                return "0";
            }
            return String(roundedValue.toFixed(0));
        }
        %>
        <div style="text-align: right;"><%- formatStatisticalValue(value) %></div>
        """
    )


def build_external_link_formatter(
    url_field: str = "external_url",
) -> HTMLTemplateFormatter:
    """Build a formatter that renders a cell value as an outbound link when available."""
    return HTMLTemplateFormatter(
        template=(
            f"<% if ({url_field}) {{ %>"
            f'<a href="<%- {url_field} %>" target="_blank" rel="noopener noreferrer"><%- value %></a>'
            "<% } else { %>"
            "<%- value %>"
            "<% } %>"
        )
    )
