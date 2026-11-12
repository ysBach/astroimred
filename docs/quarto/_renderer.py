"""Preserve multiline parameter descriptions when quartodoc renders API pages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from _scripts.normalize_api import normalize_api
from quartodoc.pandoc.blocks import DefinitionList
from quartodoc.renderers.md_renderer import MdRenderer, ParamRow


class Renderer(MdRenderer):
    """Keep parameter descriptions as Markdown blocks in definition lists."""

    style = "astroimred-markdown"

    def _render_table(
        self,
        rows: Sequence[ParamRow],
        headers: Sequence[str],
        style: Literal["parameters", "attributes", "returns"],
    ) -> str:
        """Render rows without flattening definition-list descriptions.

        Parameters
        ----------
        rows : sequence of ParamRow
            API descriptions and their signature metadata.
        headers : sequence of str
            Column names for the standard table format.
        style : {'parameters', 'attributes', 'returns'}
            Kind of API section being rendered.

        Returns
        -------
        str
            Markdown preserving paragraph, list, math, and callout structure.
        """
        if self.table_style != "description-list":
            return super()._render_table(rows, headers, style)
        return str(
            DefinitionList(
                [
                    (row.to_definition_list()[0], normalize_api(row.description))
                    for row in rows
                ]
            )
        )
