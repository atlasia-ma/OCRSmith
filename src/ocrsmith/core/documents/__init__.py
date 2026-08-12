"""Document generation: content models, page geometry, typography and flow layout."""

from .content import ContentBlock, DocumentBuilder, DocumentContent
from .flow import DocumentRenderer, RenderedPage
from .page_spec import PAPER_SIZES_MM, PageSpec
from .table_renderer import BorderStyle, RenderedTable, TableRenderer, TableStyle
from .templates import (
    ArticleTemplate,
    DocumentTemplate,
    FormTemplate,
    InvoiceTemplate,
    LetterTemplate,
    NewspaperTemplate,
    ReportTemplate,
    TemplateRegistry,
    default_registry,
)
from .text_source import CorpusTextProvider, TextProvider
from .typography import (
    FontFamily,
    RoleTypography,
    Typography,
    TypographySampler,
    group_font_families,
)

__all__ = [
    "PAPER_SIZES_MM",
    "ArticleTemplate",
    "BorderStyle",
    "ContentBlock",
    "CorpusTextProvider",
    "DocumentBuilder",
    "DocumentContent",
    "DocumentRenderer",
    "DocumentTemplate",
    "FontFamily",
    "FormTemplate",
    "InvoiceTemplate",
    "LetterTemplate",
    "NewspaperTemplate",
    "PageSpec",
    "RenderedPage",
    "RenderedTable",
    "ReportTemplate",
    "RoleTypography",
    "TableRenderer",
    "TableStyle",
    "TemplateRegistry",
    "TextProvider",
    "Typography",
    "TypographySampler",
    "default_registry",
    "group_font_families",
]
