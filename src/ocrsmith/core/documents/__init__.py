"""Document generation: content models, page geometry, typography and flow layout."""

from .charts import Chart, ChartKind, ChartRenderer, ChartSeries, sample_chart
from .content import ContentBlock, DocumentBuilder, DocumentContent
from .flow import DocumentRenderer, RenderedPage
from .formulas import FormulaRenderer, choose_math_font, sample_formula
from .page_spec import PAPER_SIZES_MM, PageSpec
from .table_renderer import BorderStyle, RenderedTable, TableRenderer, TableStyle
from .templates import (
    ArticleTemplate,
    ContentsTemplate,
    DocumentTemplate,
    FormTemplate,
    InvoiceTemplate,
    LetterTemplate,
    NewspaperTemplate,
    NotesTemplate,
    PaperTemplate,
    ReportTemplate,
    SlideTemplate,
    TemplateRegistry,
    default_registry,
)
from .text_source import CorpusTextProvider, FieldGenerator, TextProvider
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
    "Chart",
    "ContentsTemplate",
    "NotesTemplate",
    "SlideTemplate",
    "ChartKind",
    "ChartRenderer",
    "ChartSeries",
    "FormulaRenderer",
    "BorderStyle",
    "ContentBlock",
    "CorpusTextProvider",
    "DocumentBuilder",
    "DocumentContent",
    "DocumentRenderer",
    "DocumentTemplate",
    "FieldGenerator",
    "FontFamily",
    "FormTemplate",
    "InvoiceTemplate",
    "LetterTemplate",
    "NewspaperTemplate",
    "PaperTemplate",
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
    "choose_math_font",
    "default_registry",
    "sample_chart",
    "sample_formula",
    "group_font_families",
]
