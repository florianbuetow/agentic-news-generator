"""Terminal histograms of video release dates over rolling day-bucket windows.

Buckets publication dates into fixed-width windows of whole calendar days and
renders each window as a block-character terminal chart. Every window is
rolling: its newest bucket always ends on the reference date. The rendered
chart shows the oldest bucket leftmost and the newest bucket rightmost.
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field

FULL_BLOCK: str = "█"
HALF_BLOCK: str = "▄"
X_AXIS_LABEL_ROWS: int = 3
Y_AXIS_LABEL_WIDTH: int = 6
Y_AXIS_TICK_FRACTIONS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
COMPACT_DATE_FORMAT: str = "%m-%d"


class DateBucket(BaseModel):
    """One histogram bucket covering an inclusive calendar-date span."""

    start: date = Field(..., description="First calendar date the bucket covers (inclusive)")
    end: date = Field(..., description="Last calendar date the bucket covers (inclusive)")
    count: int = Field(..., ge=0, description="Videos released inside the bucket span")

    model_config = ConfigDict(frozen=True, extra="forbid")


class DateHistogram(BaseModel):
    """Bucketed release counts for one rolling window ending on the reference date."""

    title: str = Field(..., min_length=1, description="Chart title shown above the plot")
    bucket_days: int = Field(..., gt=0, description="Calendar days each bucket covers")
    buckets: list[DateBucket] = Field(..., min_length=1, description="Buckets ordered oldest to newest")
    window_start: date = Field(..., description="First calendar date of the window (inclusive)")
    window_end: date = Field(..., description="Last calendar date of the window (inclusive)")
    total_in_window: int = Field(..., ge=0, description="Videos released inside the window")
    older_count: int = Field(..., ge=0, description="Videos released before the window start")
    newer_count: int = Field(..., ge=0, description="Videos with a release date after the window end")

    model_config = ConfigDict(frozen=True, extra="forbid")


def build_date_histogram(
    release_dates: list[date],
    window_end: date,
    bucket_days: int,
    bucket_count: int,
    title: str,
) -> DateHistogram:
    """Bucket release dates into a rolling window that ends on window_end.

    Args:
        release_dates: Calendar dates the videos were released on.
        window_end: Newest date the window covers (inclusive), usually today.
        bucket_days: Calendar days each bucket covers.
        bucket_count: Number of buckets in the window.
        title: Chart title carried into the rendered output.

    Returns:
        The bucketed histogram; dates outside the window are counted as older
        or newer instead of being dropped silently.

    Raises:
        ValueError: If bucket_days or bucket_count is not positive.
    """
    if bucket_days <= 0:
        raise ValueError("bucket_days must be greater than zero")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be greater than zero")

    window_start = window_end - timedelta(days=bucket_days * bucket_count - 1)
    counts = [0] * bucket_count
    older = 0
    newer = 0
    for release_date in release_dates:
        if release_date < window_start:
            older += 1
        elif release_date > window_end:
            newer += 1
        else:
            counts[(release_date - window_start).days // bucket_days] += 1

    buckets: list[DateBucket] = []
    for index, count in enumerate(counts):
        bucket_start = window_start + timedelta(days=index * bucket_days)
        buckets.append(DateBucket(start=bucket_start, end=bucket_start + timedelta(days=bucket_days - 1), count=count))

    return DateHistogram(
        title=title,
        bucket_days=bucket_days,
        buckets=buckets,
        window_start=window_start,
        window_end=window_end,
        total_in_window=sum(counts),
        older_count=older,
        newer_count=newer,
    )


def format_bucket_label(bucket_start: date, bucket_days: int) -> str:
    """Format a bucket's start date for the x-axis.

    Daily buckets sit inside a short window, so a compact month-day label is
    enough; wider buckets can span years and keep the full ISO date.
    """
    if bucket_days == 1:
        return bucket_start.strftime(COMPACT_DATE_FORMAT)
    return bucket_start.isoformat()


def place_label(row: list[str], position: int, label: str) -> None:
    """Place a label into a row without exceeding row bounds."""
    if len(label) > len(row):
        return
    start = min(max(position, 0), len(row) - len(label))
    for offset, char in enumerate(label):
        row[start + offset] = char


def render_x_axis(histogram: DateHistogram, cell_width: int, label_stride: int) -> list[str]:
    """Render tick marks and staggered bucket start-date labels.

    Args:
        histogram: The histogram whose buckets are labelled.
        cell_width: Terminal columns each bucket occupies.
        label_stride: Label every N-th bucket, starting at the oldest.

    Returns:
        The tick row followed by the staggered label rows, gutter included.
    """
    plot_width = len(histogram.buckets) * cell_width
    gutter = " " * (Y_AXIS_LABEL_WIDTH + 2)
    tick_row = [" "] * plot_width
    label_rows = [[" "] * plot_width for _ in range(X_AXIS_LABEL_ROWS)]

    for label_number, bucket_index in enumerate(range(0, len(histogram.buckets), label_stride)):
        position = bucket_index * cell_width
        tick_row[min(position, plot_width - 1)] = "|"
        label = format_bucket_label(histogram.buckets[bucket_index].start, histogram.bucket_days)
        place_label(label_rows[label_number % X_AXIS_LABEL_ROWS], position, label)

    return [gutter + "".join(tick_row), *[(gutter + "".join(row)).rstrip() for row in label_rows]]


def render_bucket_cell(count: int, row_floor: float, row_ceiling: float, row_height: float) -> str:
    """Render one chart cell for a bucket at one count row."""
    if count >= row_ceiling:
        return FULL_BLOCK
    if count > row_floor:
        fill_fraction = (count - row_floor) / row_height
        return FULL_BLOCK if fill_fraction >= 0.5 else HALF_BLOCK
    return " "


def y_axis_ticks(height: int, max_count: int) -> dict[int, str]:
    """Map every chart row to its count label.

    Rows at the standard tick fractions carry the scaled count; every other
    row carries an empty label. Fractions are visited from the top down, so
    when two fractions round to the same row the larger fraction keeps it.
    """
    ticks: dict[int, str] = {row: "" for row in range(1, height + 1)}
    for fraction in Y_AXIS_TICK_FRACTIONS:
        row = round(fraction * height)
        if row > 0 and not ticks[row]:
            ticks[row] = f"{round(fraction * max_count):,}"
    return ticks


def render_date_histogram(histogram: DateHistogram, chart_height: int, cell_width: int, label_stride: int) -> str:
    """Render one date histogram as terminal text.

    The bar area is at most chart_height rows tall; when the tallest bucket
    holds fewer videos than that, the chart shrinks so one row equals one
    video and no false resolution is shown.

    Args:
        histogram: The bucketed release counts to draw.
        chart_height: Maximum bar-area height in terminal rows.
        cell_width: Terminal columns each bucket occupies.
        label_stride: Label every N-th bucket on the x-axis.

    Returns:
        The rendered chart including header, axes, and labels.

    Raises:
        ValueError: If chart_height, cell_width, or label_stride is not positive.
    """
    if chart_height <= 0:
        raise ValueError("chart_height must be greater than zero")
    if cell_width <= 0:
        raise ValueError("cell_width must be greater than zero")
    if label_stride <= 0:
        raise ValueError("label_stride must be greater than zero")

    window_days = histogram.bucket_days * len(histogram.buckets)
    lines: list[str] = [
        histogram.title,
        f"Videos: {histogram.total_in_window:,} | Window: {histogram.window_start.isoformat()} to {histogram.window_end.isoformat()} "
        f"({window_days} days, 1 bar = {histogram.bucket_days} day(s))",
        "",
    ]

    if histogram.total_in_window == 0:
        lines.append("No videos released in this window.")
        return "\n".join(lines)

    counts = [bucket.count for bucket in histogram.buckets]
    max_count = max(counts)
    height = min(chart_height, max_count)
    row_height = max_count / height
    plot_width = len(histogram.buckets) * cell_width
    ticks = y_axis_ticks(height, max_count)

    lines.append(f"{'count':>{Y_AXIS_LABEL_WIDTH}} |{' ' * plot_width}|")
    for row in range(height, 0, -1):
        row_floor = (row - 1) * row_height
        row_ceiling = row * row_height
        bars = "".join(render_bucket_cell(count, row_floor, row_ceiling, row_height) * cell_width for count in counts)
        lines.append(f"{ticks[row]:>{Y_AXIS_LABEL_WIDTH}} |{bars}|")
    lines.append(f"{'0':>{Y_AXIS_LABEL_WIDTH}} +{'-' * plot_width}+")
    lines.extend(render_x_axis(histogram, cell_width, label_stride))

    if histogram.newer_count > 0:
        lines.append("")
        lines.append(f"Future-dated releases after {histogram.window_end.isoformat()}: {histogram.newer_count:,} video(s)")
    return "\n".join(lines)
