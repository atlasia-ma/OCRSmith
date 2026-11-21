"""Document templates.

Each template describes one *genre* of document. Genre matters more than it looks: a model
trained only on flowing articles never learns that a receipt has right-aligned amounts,
that a form has label/value pairs, or that a newspaper's columns are read across a
gutter. The registry lets a generation run sample genres by weight, so a corpus can be
deliberately balanced instead of accidentally uniform.

Templates produce content only. Page size, fonts and degradation are decided elsewhere,
which is what lets the same article appear as a crisp A4 print and a creased phone photo
with identical markup ground truth.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .charts import ChartKind, sample_chart
from .content import DocumentBuilder, DocumentContent
from .formulas import sample_formula
from .page_spec import PageProfile
from .text_source import FieldGenerator, TextProvider

__all__ = [
    "DEFAULT_PROFILE",
    "PROFILES",
    "profile_for",
    "scale_count",
    "ArticleTemplate",
    "ContentsTemplate",
    "NotesTemplate",
    "SlideTemplate",
    "DocumentTemplate",
    "FormTemplate",
    "InvoiceTemplate",
    "LetterTemplate",
    "BusinessCardTemplate",
    "LabelTemplate",
    "MagazineTemplate",
    "NewspaperTemplate",
    "ReceiptTemplate",
    "PaperTemplate",
    "ReportTemplate",
    "TemplateRegistry",
    "default_registry",
]


def scale_count(
    options: dict,
    rng: random.Random,
    *,
    base: tuple[int, int],
    per_1000_chars: float,
    cap: int,
) -> int:
    """How many of a repeated element to emit, given how much room the sheet has.

    `size_hint` is the chosen page's capacity in characters. A template that ignores it
    puts the same five paragraphs on a broadsheet and on an A5, which fills one and leaves
    the other nine tenths blank - the page size was being decided by the document type and
    then not being answered by the content. Absent a hint, the base range still applies, so
    a caller that knows nothing about the page gets sensible output.
    """
    low, high = base
    hint = options.get("size_hint")
    count = rng.randint(low, high)
    if hint:
        count = max(count, int(round(hint / 1000.0 * per_1000_chars)))
    return max(1, min(cap, count))


@runtime_checkable
class DocumentTemplate(Protocol):
    """Builds one document's content from a text source."""

    name: str

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent: ...


@dataclass(frozen=True, slots=True)
class ArticleTemplate:
    """Title, lead paragraph, then sections of prose — the commonest document shape."""

    name: str = "article"
    min_sections: int = 2
    max_sections: int = 5

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        direction = options.get("direction")
        builder = DocumentBuilder(direction, template=self.name)
        builder.title(source.title(rng))
        builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
        for _ in range(
            scale_count(
                options,
                rng,
                base=(self.min_sections, self.max_sections),
                per_1000_chars=1.6,
                cap=12,
            )
        ):
            builder.heading(source.title(rng))
            for _ in range(rng.randint(1, 3)):
                builder.paragraph(source.paragraph(rng, rng.randint(2, 5)))
            if rng.random() < 0.25:
                builder.list([source.phrase(rng, 6) for _ in range(rng.randint(2, 5))])
        return builder.build()


@dataclass(frozen=True, slots=True)
class ReportTemplate:
    """Headed report with a data table, a figure and a caption."""

    name: str = "report"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.header(source.phrase(rng, 4))
        builder.title(source.title(rng))
        builder.paragraph(source.paragraph(rng, 3))
        builder.heading(source.title(rng))
        builder.table(_sample_table(source, rng))
        builder.caption(source.phrase(rng, 6))
        builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
        if rng.random() < 0.55:
            labels = [source.phrase(rng, 1) for _ in range(4)]
            builder.chart(
                sample_chart(rng, labels, title=source.phrase(rng, 3)),
                width=rng.randint(260, 420),
                height=rng.randint(190, 300),
            )
            builder.caption(source.phrase(rng, 5))
        elif rng.random() < 0.5:
            builder.figure(width=rng.randint(240, 420), height=rng.randint(160, 300))
            builder.caption(source.phrase(rng, 5))
        for _ in range(scale_count(options, rng, base=(1, 3), per_1000_chars=1.1, cap=14)):
            builder.paragraph(source.paragraph(rng, rng.randint(3, 5)))
        return builder.build()


@dataclass(frozen=True, slots=True)
class LetterTemplate:
    """Correspondence: addressee, body, sign-off."""

    name: str = "letter"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.paragraph(source.phrase(rng, 5))
        builder.separator()
        builder.paragraph(source.phrase(rng, 4))
        for _ in range(scale_count(options, rng, base=(3, 6), per_1000_chars=2.6, cap=14)):
            builder.paragraph(source.paragraph(rng, rng.randint(3, 6)))
        builder.paragraph(source.phrase(rng, 3))
        return builder.build()


@dataclass(frozen=True, slots=True)
class FormTemplate:
    """Label/value rows, the shape used by administrative forms and IDs.

    A form's labels are the most fixed text in any document: every administrative form in
    Morocco asks for a national identity number under the words "رقم البطاقة الوطنية". They
    were being drawn from the prose corpus, which produced fields captioned "والقاف" and
    "الرقعة" - grammatically Arabic, and asking for nothing. Worse, the label and its value
    were sampled independently, so a field labelled "date" could hold a phone number.

    The pairs below are therefore named, and each one knows what *kind* of value belongs
    under it, so the document asks real questions and answers them in the right units.
    """

    name: str = "form"

    _TITLES = (
        "مطبوع طلب",
        "استمارة التسجيل",
        "طلب الحصول على وثيقة",
        "استمارة معلومات",
        "طلب فتح ملف",
    )
    _SECTIONS = ("معلومات شخصية", "معلومات الاتصال", "معطيات الملف", "معلومات إضافية")

    #: Fields grouped as a real form groups them, so a section heading describes what is
    #: under it. Shuffling one flat pool put a delivery date under "personal information".
    #: Each entry is (label, kind); the kind picks the generator, so the answer fits the
    #: question and a field labelled "date of birth" cannot come back holding a phone number.
    _GROUPS = (
        (
            "معلومات شخصية",
            (
                ("الاسم العائلي", "family_name"),
                ("الاسم الشخصي", "first_name"),
                ("رقم البطاقة الوطنية", "code"),
                ("تاريخ الازدياد", "date"),
                ("مكان الازدياد", "city"),
                ("الجنسية", "nationality"),
                ("الحالة العائلية", "marital"),
                ("المهنة", "profession"),
            ),
        ),
        (
            "معلومات الاتصال",
            (
                ("العنوان", "address"),
                ("المدينة", "city"),
                ("الرمز البريدي", "postcode"),
                ("الهاتف", "phone"),
            ),
        ),
        (
            "معطيات الملف",
            (
                ("رقم الملف", "code"),
                ("تاريخ الإيداع", "date"),
                ("المبلغ المؤدى", "money"),
                ("رقم الحساب البنكي", "code"),
                ("تاريخ التسليم", "date"),
                ("عدد الوثائق", "count"),
                ("مدة الصلاحية", "duration"),
                ("نسبة الإنجاز", "percentage"),
            ),
        ),
        (
            "معلومات إضافية",
            (
                ("الجهة", "city"),
                ("رقم الطلب", "code"),
                ("نوع الوثيقة", "doctype"),
                ("عدد النسخ", "count"),
                ("رسوم الملف", "money"),
                ("تاريخ آخر تحيين", "date"),
                ("مرجع الأداء", "code"),
                ("ملاحظات", "note"),
            ),
        ),
    )
    _DOCTYPES = (
        "شهادة إدارية",
        "نسخة من عقد",
        "شهادة السكنى",
        "بطاقة التعريف",
        "شهادة الحياة",
        "عقد ازدياد",
    )
    _FIRST_NAMES = (
        "محمد",
        "أحمد",
        "يوسف",
        "خديجة",
        "فاطمة",
        "سعيد",
        "نادية",
        "رشيد",
        "سميرة",
        "عبد الله",
        "ليلى",
        "كريم",
    )
    _FAMILY_NAMES = (
        "العلوي",
        "بنعيسى",
        "الإدريسي",
        "الفاسي",
        "بوعزة",
        "المرابط",
        "الشرقاوي",
        "بنكيران",
        "الحسني",
        "أمزيل",
        "الزهراوي",
        "بلقاسم",
    )
    _CITIES = (
        "الرباط",
        "الدار البيضاء",
        "فاس",
        "مراكش",
        "طنجة",
        "أكادير",
        "وجدة",
        "مكناس",
        "تطوان",
        "القنيطرة",
        "سلا",
        "الجديدة",
    )
    _PROFESSIONS = (
        "أستاذ",
        "مهندس",
        "طبيب",
        "تاجر",
        "موظف",
        "محاسب",
        "تقني",
        "صحفي",
        "مترجم",
        "حرفي",
    )
    _STREETS = ("شارع الحسن الثاني", "زنقة الأطلس", "شارع محمد الخامس", "زنقة النخيل", "حي السلام")
    _MARITAL = ("أعزب", "متزوج", "مطلق", "أرمل")

    def _value(self, kind: str, source: TextProvider, fields: FieldGenerator, rng, unit: str) -> str:
        if kind == "code":
            return fields.code(rng)
        if kind == "date":
            return fields.date(rng)
        if kind == "phone":
            return fields.phone(rng)
        if kind == "money":
            return fields.amount(rng, unit)
        if kind == "percentage":
            return fields.percentage(rng)
        if kind == "city":
            return rng.choice(self._CITIES)
        if kind == "profession":
            return rng.choice(self._PROFESSIONS)
        if kind == "marital":
            return rng.choice(self._MARITAL)
        if kind == "nationality":
            return "مغربية"
        if kind == "first_name":
            return rng.choice(self._FIRST_NAMES)
        if kind == "family_name":
            return rng.choice(self._FAMILY_NAMES)
        if kind == "doctype":
            return rng.choice(self._DOCTYPES)
        if kind == "note":
            return source.phrase(rng, 4)
        if kind == "postcode":
            # No thousands separator: a postcode is an identifier, not a quantity.
            return fields.digits(str(rng.randint(10000, 99999)))
        if kind == "count":
            return fields.plain_number(rng.randint(1, 40))
        if kind == "duration":
            return f"{fields.plain_number(rng.randint(1, 24))} شهرا"
        if kind == "address":
            return f"{rng.choice(self._STREETS)}، {fields.plain_number(rng.randint(1, 320))}"
        return source.phrase(rng, 2)

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        fields = FieldGenerator(options.get("numerals", "keep"))
        # One unit for the document, as on the invoice: a form that quotes a fee in euros
        # and a deposit in dirhams is quoting two different documents.
        unit = fields.currency(rng)
        builder.title(rng.choice(self._TITLES))

        wanted = scale_count(options, rng, base=(12, 20), per_1000_chars=7.0, cap=28)
        first = True
        for heading, group in self._GROUPS:
            if wanted <= 0:
                break
            take = min(len(group), max(3, wanted))
            chosen = list(group[:take])
            wanted -= take
            if not first:
                builder.separator()
                builder.heading(heading)
            builder.key_values(
                [(label, self._value(kind, source, fields, rng, unit)) for label, kind in chosen]
            )
            first = False

        # Every administrative form ends by asking who filled it in and when, with room
        # left for a hand. Without it the sheet simply stopped, half used.
        builder.separator()
        builder.key_values(
            [
                ("حرر بـ", rng.choice(self._CITIES)),
                ("بتاريخ", fields.date(rng)),
                ("الاسم والصفة", f"{rng.choice(self._FIRST_NAMES)} {rng.choice(self._FAMILY_NAMES)}"),
                ("التوقيع", ""),
            ]
        )
        return builder.build()


@dataclass(frozen=True, slots=True)
class InvoiceTemplate:
    """Header, line-item table with totals, then terms."""

    name: str = "invoice"

    #: Real column headings, not sampled prose. An invoice's table is the one part of it
    #: that is not free text: every invoice in the world labels a quantity column
    #: "quantity". Drawing those labels from the corpus produced headers that read as a
    #: sentence fragment, and a literal "#" sat in the quantity column because nothing had
    #: given it a word.
    _HEADERS = ("البيان", "الكمية", "ثمن الوحدة", "المجموع")
    _META = ("رقم الفاتورة", "التاريخ", "الهاتف", "الرقم الضريبي")
    _TOTALS = ("المجموع الفرعي", "الضريبة على القيمة المضافة", "المجموع الإجمالي")
    _TITLES = ("فاتورة", "فاتورة أداء", "فاتورة مبيعات", "إشعار بالأداء", "فاتورة الخدمات")

    #: Things an invoice can actually be for. Line items were being drawn from the prose
    #: corpus, which produced entries like "نشأ في الدواوين" and "لا يحسب" - sentence
    #: fragments priced per unit. A line item is a short noun phrase naming a good or a
    #: service, and no corpus of continuous prose contains those, so they are named here.
    _ITEMS = (
        "خدمات استشارية",
        "تصميم موقع إلكتروني",
        "استضافة سنوية",
        "ترخيص برمجي",
        "صيانة الأجهزة",
        "طباعة وثائق",
        "تجهيزات مكتبية",
        "ورق تصوير",
        "حبر الطابعة",
        "خدمات الترجمة",
        "تكوين الموظفين",
        "دراسة تقنية",
        "شحن ونقل",
        "أشغال التركيب",
        "لوازم كهربائية",
        "اشتراك الإنترنت",
        "خدمات المحاسبة",
        "تدقيق الحسابات",
        "كراء القاعة",
        "تأمين المعدات",
        "قطع الغيار",
        "خدمات التنظيف",
        "حراسة وأمن",
        "استشارة قانونية",
    )

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        fields = FieldGenerator(options.get("numerals", "keep"))
        # One unit for the whole document. An invoice does not change currency halfway
        # down its own item table.
        unit = fields.currency(rng)

        builder.title(rng.choice(self._TITLES))
        builder.key_values(
            [
                (self._META[0], fields.code(rng)),
                (self._META[1], fields.date(rng)),
                (self._META[2], fields.phone(rng)),
                (self._META[3], fields.code(rng)),
            ]
        )

        # An invoice is mostly its item table, so the table is what grows to fill the
        # sheet. `size_hint` is the sheet's capacity in characters, measured by the caller
        # from the page it has already chosen - without it a five-line invoice landed in
        # the top corner of an A4 and left the rest of the sheet empty.
        rows = [list(self._HEADERS)]
        subtotal = 0.0
        for _ in range(scale_count(options, rng, base=(4, 9), per_1000_chars=3.5, cap=26)):
            quantity = rng.randint(1, 40)
            unit_price = rng.uniform(5, 4000)
            line_total = quantity * unit_price
            subtotal += line_total
            rows.append(
                [
                    rng.choice(self._ITEMS),
                    fields.plain_number(quantity),
                    fields.money(rng, unit_price, unit),
                    fields.money(rng, line_total, unit),
                ]
            )
        builder.table(rows)

        # The totals are the sum of the lines above them. Drawing them independently left a
        # subtotal of 39,864 under a table of items adding to 437,000 - every number on the
        # page individually plausible and the document as a whole impossible. A model
        # learning to read invoices should meet arithmetic that holds.
        vat = subtotal * 0.20
        builder.key_values(
            [
                (self._TOTALS[0], fields.money(rng, subtotal, unit)),
                (self._TOTALS[1], fields.money(rng, vat, unit)),
                (self._TOTALS[2], fields.money(rng, subtotal + vat, unit)),
            ]
        )
        builder.paragraph(source.paragraph(rng, 2))
        return builder.build()


@dataclass(frozen=True, slots=True)
class PaperTemplate:
    """An academic paper: prose interleaved with displayed equations.

    Formula conversion is the largest gain category in the document-parsing benchmarks,
    and it needs formulas *in context* - a page of nothing but equations is not what a
    model meets in the wild.
    """

    name: str = "paper"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.title(rng))
        builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
        for _ in range(rng.randint(2, 4)):
            builder.heading(source.title(rng))
            builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
            builder.formula(sample_formula(rng))
            if rng.random() < 0.4:
                builder.paragraph(source.paragraph(rng, 2))
        if rng.random() < 0.5:
            builder.chart(
                sample_chart(
                    rng,
                    [source.phrase(rng, 1) for _ in range(4)],
                    title=source.phrase(rng, 3),
                    kind=ChartKind.LINE,
                ),
                width=rng.randint(240, 400),
                height=rng.randint(170, 260),
            )
            builder.caption(source.phrase(rng, 5))
        return builder.build()


@dataclass(frozen=True, slots=True)
class NewspaperTemplate:
    """Dense multi-column prose with a masthead and short headlines."""

    name: str = "newspaper"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.header(source.phrase(rng, 3))
        builder.title(source.title(rng))
        # A newspaper's whole point is that its columns are full. `size_hint` already counts
        # every column of the chosen sheet, so scaling against it is what keeps a five-column
        # broadsheet from running two columns of copy and leaving the rest of the sheet
        # blank - which reads as full on any measure of vertical fill and is still half an
        # empty page.
        for index in range(scale_count(options, rng, base=(4, 8), per_1000_chars=0.9, cap=30)):
            if index:
                builder.heading(source.title(rng), level=3)
            for _ in range(rng.randint(2, 4)):
                builder.paragraph(source.paragraph(rng, rng.randint(3, 6)))
        return builder.build()


@dataclass(frozen=True, slots=True)
class ContentsTemplate:
    """A table of contents: label, dot leaders, page number.

    Dot leaders are their own recognition problem - a long run of identical glyphs that
    models routinely miscount or hallucinate - and no synthetic corpus generates them.
    """

    name: str = "contents"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.phrase(rng, 2))
        page = rng.randint(1, 9)
        for _ in range(scale_count(options, rng, base=(10, 18), per_1000_chars=5.0, cap=34)):
            label = source.phrase(rng, rng.randint(2, 5))
            leader = "." * rng.randint(6, 30)
            builder.paragraph(f"{label} {leader} {page}")
            page += rng.randint(1, 12)
        return builder.build()


@dataclass(frozen=True, slots=True)
class SlideTemplate:
    """A presentation slide: a headline and a few short bullets, set large.

    Slides are a distinct visual regime - very large type, very little of it - and a model
    trained only on dense prose reads them poorly.
    """

    name: str = "slide"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.phrase(rng, rng.randint(2, 5)))
        builder.list([source.phrase(rng, rng.randint(3, 8)) for _ in range(rng.randint(3, 6))])
        if rng.random() < 0.4:
            builder.chart(
                sample_chart(rng, [source.phrase(rng, 1) for _ in range(4)], title=""),
                width=rng.randint(280, 420),
                height=rng.randint(180, 260),
            )
        builder.footer(source.phrase(rng, 2))
        return builder.build()


@dataclass(frozen=True, slots=True)
class NotesTemplate:
    """Handwritten notes: a heading and loose lines, drawn in a handwriting face.

    The Arabic benchmarks are handwriting-heavy (KHATT, Muharaf, and most of KITAB-Bench's
    handwritten domain), and a printed-only corpus transfers to them poorly. This is not a
    substitute for real handwriting data - the letterforms come from a font, so the
    variability of a human hand is missing - but it covers the layout and the visual
    regime, and it is honest about which it is.
    """

    name: str = "notes"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name, handwritten=True)
        builder.heading(source.phrase(rng, rng.randint(2, 4)))
        for _ in range(scale_count(options, rng, base=(6, 12), per_1000_chars=4.5, cap=26)):
            builder.paragraph(source.sentence(rng))
        if rng.random() < 0.5:
            builder.list([source.phrase(rng, rng.randint(2, 6)) for _ in range(rng.randint(2, 5))])
        return builder.build()


@dataclass(frozen=True, slots=True)
class ReceiptTemplate:
    """A till roll: merchant, items, totals, payment, footer line.

    Structurally the furthest thing from prose in the whole registry, and the shape the
    reference set has a whole class for. Everything is one narrow column, amounts range to
    the opposite edge of their labels, and there is no continuous text anywhere on it.
    """

    name: str = "receipt"

    _SHOPS = (
        "بقالة الأمل",
        "سوبر ماركت النجاح",
        "مخبزة الرياض",
        "صيدلية الشفاء",
        "مقهى الأصدقاء",
        "محل الأدوات المنزلية",
        "مكتبة المعرفة",
    )
    _ITEMS = (
        "خبز",
        "حليب",
        "زيت الزيتون",
        "سكر",
        "أرز",
        "شاي أخضر",
        "قهوة",
        "بيض",
        "جبن",
        "طماطم",
        "بصل",
        "دجاج",
        "ماء معدني",
        "عصير برتقال",
        "منظف",
        "صابون",
        "ورق مطبخ",
        "معجون أسنان",
    )
    _PAYMENT = ("نقدا", "بطاقة بنكية", "تحويل")
    _FOOTERS = ("شكرا لزيارتكم", "نتمنى لكم يوما سعيدا", "البضاعة المباعة لا ترد", "إلى اللقاء")

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        fields = FieldGenerator(options.get("numerals", "keep"))
        unit = rng.choice(("درهم", "DH", "MAD"))

        builder.title(rng.choice(self._SHOPS))
        builder.paragraph(f"{rng.choice(FormTemplate._STREETS)}، {rng.choice(FormTemplate._CITIES)}")
        builder.key_values(
            [
                ("التاريخ", fields.date(rng)),
                ("الساعة", f"{rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}"),
                ("رقم الوصل", fields.code(rng)),
            ]
        )
        builder.separator()

        total = 0.0
        pairs = []
        for _ in range(scale_count(options, rng, base=(5, 12), per_1000_chars=6.0, cap=26)):
            quantity = rng.randint(1, 6)
            # Grocery prices, not invoice prices. A till roll listing rice at 98 dirhams a
            # kilo is arithmetically consistent and still not a shopping basket.
            price = rng.uniform(2.5, 65.0)
            line = quantity * price
            total += line
            pairs.append((f"{rng.choice(self._ITEMS)} x{quantity}", fields.money(rng, line, unit)))
        builder.key_values(pairs)
        builder.separator()

        vat = total * 0.20
        builder.key_values(
            [
                ("المجموع", fields.money(rng, total, unit)),
                ("الضريبة", fields.money(rng, vat, unit)),
                ("المبلغ المؤدى", fields.money(rng, total + vat, unit)),
                ("طريقة الأداء", rng.choice(self._PAYMENT)),
            ]
        )
        builder.paragraph(rng.choice(self._FOOTERS))
        return builder.build()


@dataclass(frozen=True, slots=True)
class BusinessCardTemplate:
    """Name, role, organisation, and how to reach them. Six to ten lines, no prose."""

    name: str = "business_card"

    _ROLES = (
        "مدير عام",
        "مهندس معلوميات",
        "مستشار قانوني",
        "مسؤول التواصل",
        "خبير محاسب",
        "مديرة الموارد البشرية",
        "رئيس قسم المبيعات",
        "مهندسة معمارية",
    )
    _ORGS = (
        "شركة الأطلس للخدمات",
        "مجموعة النور",
        "مكتب الدراسات التقنية",
        "مؤسسة الرواد",
        "شركة البحر الأبيض",
    )

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        fields = FieldGenerator(options.get("numerals", "keep"))
        person = f"{rng.choice(FormTemplate._FIRST_NAMES)} {rng.choice(FormTemplate._FAMILY_NAMES)}"

        builder.title(person)
        builder.paragraph(rng.choice(self._ROLES))
        builder.heading(rng.choice(self._ORGS), level=3)
        builder.key_values(
            [
                ("الهاتف", fields.phone(rng)),
                ("العنوان", f"{rng.choice(FormTemplate._STREETS)}، {rng.choice(FormTemplate._CITIES)}"),
            ]
        )
        return builder.build()


@dataclass(frozen=True, slots=True)
class LabelTemplate:
    """A product or shelf label: what it is, its code, and its numbers."""

    name: str = "label"

    _PRODUCTS = (
        "زيت الزيتون البكر",
        "عسل طبيعي",
        "تمر المجهول",
        "شاي أخضر بالنعناع",
        "أرغان تجميلي",
        "قهوة مطحونة",
        "مربى المشمش",
        "توابل مشكلة",
    )
    _UNITS = ("500 غ", "1 كغ", "250 مل", "750 مل", "2 كغ", "100 غ")

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        fields = FieldGenerator(options.get("numerals", "keep"))
        unit = rng.choice(("درهم", "DH"))

        builder.title(rng.choice(self._PRODUCTS))
        builder.key_values(
            [
                ("الوزن", rng.choice(self._UNITS)),
                ("الثمن", fields.money(rng, rng.uniform(8, 320), unit)),
                ("رقم التتبع", fields.code(rng)),
                ("تاريخ الإنتاج", fields.date(rng)),
                ("صالح إلى", fields.date(rng)),
            ]
        )
        return builder.build()


@dataclass(frozen=True, slots=True)
class MagazineTemplate:
    """A feature spread: standfirst, sections, and a pull quote."""

    name: str = "magazine"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.title(rng))
        # The standfirst: a short bold intro that is neither the headline nor body copy.
        builder.quote(source.paragraph(rng, 2))
        for index in range(scale_count(options, rng, base=(3, 6), per_1000_chars=1.3, cap=16)):
            if index:
                builder.heading(source.title(rng), level=3)
            for _ in range(rng.randint(2, 4)):
                builder.paragraph(source.paragraph(rng, rng.randint(3, 5)))
            if index and rng.random() < 0.3:
                builder.quote(source.phrase(rng, 9))
        return builder.build()


def _sample_table(source: TextProvider, rng: random.Random) -> list[list[str]]:
    cols = rng.randint(2, 5)
    rows = rng.randint(3, 7)
    header = [source.phrase(rng, 2) for _ in range(cols)]
    body = [
        [source.phrase(rng, 2) if col == 0 else str(rng.randint(0, 9999)) for col in range(cols)]
        for _ in range(rows)
    ]
    return [header, *body]


class TemplateRegistry:
    """Named templates with sampling weights."""

    def __init__(self):
        self._templates: dict[str, DocumentTemplate] = {}
        self._weights: dict[str, float] = {}

    def register(self, template: DocumentTemplate, weight: float = 1.0) -> TemplateRegistry:
        self._templates[template.name] = template
        self._weights[template.name] = max(0.0, weight)
        return self

    def get(self, name: str) -> DocumentTemplate:
        try:
            return self._templates[name]
        except KeyError:
            raise ValueError(
                f"Unknown template {name!r}. Available: {', '.join(sorted(self._templates))}"
            ) from None

    def sample(self, rng: random.Random, names: list[str] | None = None) -> DocumentTemplate:
        candidates = names or list(self._templates)
        weights = [self._weights.get(name, 1.0) for name in candidates]
        if not candidates or sum(weights) <= 0:
            raise ValueError("No templates available to sample from")
        return self._templates[rng.choices(candidates, weights=weights, k=1)[0]]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def __len__(self) -> int:
        return len(self._templates)

    def __iter__(self) -> Iterator[DocumentTemplate]:
        return iter(self._templates.values())

    def __contains__(self, name: object) -> bool:
        return name in self._templates


#: What each document type physically is. Drawn from the class breakdown of the Humans in
#: the Loop Arabic document set (invoice, receipt, official document, admin form, book,
#: newspaper, magazine, business card, label), which is real photographed paper rather than
#: anything generated - so these are conventions observed, not invented.
#:
#: The point of stating them per type is that they constrain each other. A receipt is an
#: 80 mm roll *and* one column *and* 7-9 point type *and* short lines, and no one of those
#: is a free choice once the others are made.
PROFILES: dict[str, PageProfile] = {
    # Prose. A book page is a single justified column with generous margins; the type is
    # small because the measure is narrow, not because the sheet is.
    "article": PageProfile(
        papers={"a4": 3.0, "a5": 2.0, "b5": 1.5, "letter": 1.0},
        columns={1: 5.0, 2: 1.0},
        margin_ratio=(0.08, 0.14),
        body_pt=(9.5, 12.0),
        footer_probability=0.55,
        header_probability=0.3,
    ),
    "report": PageProfile(
        papers={"a4": 5.0, "letter": 2.0},
        columns={1: 4.0, 2: 1.5},
        margin_ratio=(0.07, 0.12),
        body_pt=(9.5, 11.5),
        header_probability=0.5,
        footer_probability=0.6,
    ),
    "paper": PageProfile(
        papers={"a4": 4.0, "letter": 2.0},
        columns={2: 4.0, 1: 1.0},
        margin_ratio=(0.06, 0.10),
        body_pt=(8.5, 10.5),
        footer_probability=0.5,
    ),
    "contents": PageProfile(
        papers={"a4": 3.0, "a5": 2.0, "b5": 1.0},
        columns={1: 1.0},
        margin_ratio=(0.09, 0.15),
        body_pt=(9.5, 12.0),
        footer_probability=0.4,
    ),
    # Correspondence. One column always - a letter in two columns is not a letter.
    "letter": PageProfile(
        papers={"a4": 5.0, "letter": 2.0},
        columns={1: 1.0},
        margin_ratio=(0.09, 0.15),
        body_pt=(10.0, 12.0),
        header_probability=0.6,
    ),
    # Transactional. The item table is the document; margins are tight because the paper
    # is doing a job rather than being read for pleasure.
    "invoice": PageProfile(
        papers={"a4": 6.0, "letter": 2.0},
        columns={1: 1.0},
        margin_ratio=(0.05, 0.09),
        body_pt=(8.5, 11.0),
        header_probability=0.75,
        footer_probability=0.4,
    ),
    # A till roll. 80 mm wide, one column, small type, and never anything else.
    "receipt": PageProfile(
        papers={"receipt": 1.0},
        columns={1: 1.0},
        margin_ratio=(0.04, 0.08),
        body_pt=(7.0, 9.5),
        leading=(1.0, 1.3),
    ),
    # Forms are mostly rules and labels, set tight, with room left for a hand to write.
    "form": PageProfile(
        papers={"a4": 6.0, "letter": 2.0, "a5": 1.0},
        columns={1: 1.0},
        margin_ratio=(0.05, 0.10),
        body_pt=(9.0, 11.5),
        header_probability=0.7,
        leading=(1.15, 1.45),
    ),
    # Dense, many-column, small type. The one type where a big sheet is the point.
    "newspaper": PageProfile(
        papers={"broadsheet": 3.0, "tabloid": 2.5, "a3": 1.5},
        columns={4: 3.0, 5: 2.0, 3: 2.0, 6: 1.0},
        margin_ratio=(0.03, 0.06),
        body_pt=(7.5, 9.5),
        header_probability=0.5,
        leading=(0.9, 1.05),
    ),
    "magazine": PageProfile(
        papers={"a4": 3.0, "tabloid": 1.0, "letter": 1.0},
        columns={2: 3.0, 3: 2.0, 1: 1.0},
        margin_ratio=(0.05, 0.10),
        body_pt=(8.5, 10.5),
        footer_probability=0.4,
    ),
    # A card is a handful of lines set large relative to a very small sheet.
    "business_card": PageProfile(
        papers={"business_card": 1.0},
        columns={1: 1.0},
        margin_ratio=(0.06, 0.12),
        body_pt=(6.5, 8.5),
        landscape_probability=0.85,
        leading=(1.1, 1.4),
    ),
    "label": PageProfile(
        papers={"id_card": 2.0, "a6": 1.5, "business_card": 1.0},
        columns={1: 1.0},
        margin_ratio=(0.05, 0.10),
        body_pt=(6.5, 9.0),
        landscape_probability=0.5,
    ),
    "slide": PageProfile(
        papers={"a4": 1.0, "letter": 1.0},
        columns={1: 1.0},
        margin_ratio=(0.07, 0.12),
        body_pt=(13.0, 18.0),
        landscape_probability=1.0,
        leading=(1.15, 1.45),
    ),
    "notes": PageProfile(
        papers={"a5": 3.0, "a4": 2.0, "a6": 1.0},
        columns={1: 1.0},
        margin_ratio=(0.06, 0.12),
        body_pt=(9.0, 12.5),
        leading=(1.15, 1.5),
    ),
}

#: Used when a template has no profile of its own.
DEFAULT_PROFILE = PROFILES["article"]


def profile_for(name: str) -> PageProfile:
    """The page profile for a template name, falling back to prose conventions."""
    return PROFILES.get(name, DEFAULT_PROFILE)


def default_registry() -> TemplateRegistry:
    """The built-in genres, weighted towards the shapes that dominate real corpora."""
    return (
        TemplateRegistry()
        .register(ArticleTemplate(), weight=3.0)
        .register(ReportTemplate(), weight=2.0)
        .register(NewspaperTemplate(), weight=1.5)
        .register(PaperTemplate(), weight=1.5)
        .register(LetterTemplate(), weight=1.0)
        .register(FormTemplate(), weight=1.0)
        .register(InvoiceTemplate(), weight=1.0)
        .register(ContentsTemplate(), weight=0.8)
        .register(SlideTemplate(), weight=0.8)
        .register(NotesTemplate(), weight=1.0)
        .register(ReceiptTemplate(), weight=1.5)
        .register(BusinessCardTemplate(), weight=0.6)
        .register(LabelTemplate(), weight=0.6)
        .register(MagazineTemplate(), weight=1.2)
    )
