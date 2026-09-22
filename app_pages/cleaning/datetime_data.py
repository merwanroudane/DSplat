import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.code_lab import code_lab
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.datasets import daily_sales, transactions
from utils.plotting import plot

page_header("datetime_data")
ts = daily_sales()

st.markdown("## التحويل Parsing والصيغ Formatting")
examples = pd.DataFrame({"raw": ["2024-03-05", "05/03/2024", "03/05/2024", "2024/3/5", "5 March 2024", "2024-02-30", ""]})
fmt = st.selectbox("الصيغة المفترضة", ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d %B %Y"], key="dt_fmt")
examples["parsed"] = pd.to_datetime(examples["raw"], format=fmt, errors="coerce")
st.dataframe(examples, hide_index=True)
st.markdown("`05/03/2024` و`03/05/2024`: نفس النص قد يعني 5 مارس أو 3 مايو. **لا توجد طريقة إحصائية لحسم ذلك** — "
            "الصيغة تأتي من مصدر البيانات. و`2024-02-30` تاريخ مستحيل يصبح NaT.")
why("حدد format صراحة في pd.to_datetime.",
    "الاستنتاج التلقائي قد يقرأ نصف العمود يوم/شهر والنصف الآخر شهر/يوم دون تحذير، فتُنتج تواريخ صالحة شكلًا وخاطئة معنى.")
st.code('df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")\n'
        'print(df["date"].isna().sum(), "values failed to parse")\n'
        'df["date"].dt.strftime("%d/%m/%Y")   # formatting for display only', language="python")

st.markdown("## استخراج المكونات · Calendar features")
parts = pd.DataFrame({"date": ts["date"].head(8)})
for name, fn in {"year": lambda d: d.dt.year, "quarter": lambda d: d.dt.quarter, "month": lambda d: d.dt.month,
                 "week": lambda d: d.dt.isocalendar().week.astype(int), "day": lambda d: d.dt.day,
                 "dayofweek": lambda d: d.dt.dayofweek, "day_name": lambda d: d.dt.day_name(),
                 "is_month_end": lambda d: d.dt.is_month_end}.items():
    parts[name] = fn(parts["date"])
st.dataframe(parts, hide_index=True)
st.caption("dayofweek: الاثنين = 0 … الأحد = 6. عطلة نهاية الأسبوع تختلف بين الدول (الجمعة/السبت أو السبت/الأحد) — قاعدة مجال.")

st.markdown("## المناطق الزمنية Time zones")
t = pd.Timestamp("2024-03-31 01:30")
zones = ["UTC", "Africa/Casablanca", "Africa/Cairo", "Asia/Riyadh", "Europe/Paris"]
tz_rows = [{"zone": z, "local time": t.tz_localize("UTC").tz_convert(z).strftime("%Y-%m-%d %H:%M %Z")} for z in zones]
st.dataframe(pd.DataFrame(tz_rows), hide_index=True)
st.markdown("خزّن الوقت بـ**UTC** مع عمود المنطقة، وحوّل محليًا عند تحليل السلوك (ساعة الذروة المحلية). "
            "التوقيت الصيفي DST يُنتج ساعات مكررة أو مفقودة.")

st.markdown("## الطوابع المفقودة والتوقيت غير المنتظم")
full = pd.date_range(ts["date"].min(), ts["date"].max(), freq="D")
missing = full.difference(ts["date"])
c1, c2, c3 = st.columns(3)
c1.metric("صفوف في الجدول", len(ts))
c2.metric("أيام متوقعة", len(full))
c3.metric("أيام غائبة", len(missing))
st.code('full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")\n'
        'df = df.set_index("date").reindex(full)   # missing days become explicit NaN rows', language="python")
tr = transactions()
gaps = tr.sort_values("datetime")["datetime"].diff().dt.total_seconds() / 3600
st.markdown(f"**بيانات أحداث غير منتظمة:** المعاملات لا تأتي بفواصل ثابتة؛ الفجوة الوسيطة بين معاملتين "
            f"{gaps.median():.1f} ساعة وأكبر فجوة {gaps.max():.0f} ساعة. نستخدم Resampling لتحويلها لسلسلة منتظمة.")

st.markdown("## Resampling")
rule = st.segmented_control("التكرار", ["D", "W", "MS", "QS"], default="W", key="dt_rule",
                            format_func={"D": "يومي", "W": "أسبوعي", "MS": "شهري", "QS": "ربع سنوي"}.get)
agg = st.radio("دالة التجميع", ["sum", "mean", "max"], horizontal=True, key="dt_agg")
rs = ts.set_index("date")["sales"].resample(rule or "W").agg(agg)
fig = go.Figure(go.Scatter(x=rs.index, y=rs, mode="lines+markers", line=dict(color=PALETTE["purple"])))
fig.update_layout(title=f"sales.resample('{rule}').{agg}()", height=320)
plot(fig)
st.caption("sum مع أيام مفقودة يعطي مجموعًا أصغر زيفًا؛ mean أقل تأثرًا. اختر الدالة حسب معنى المتغير.")

st.markdown("## Lag وLead وRolling windows")
s = ts.set_index("date")["sales"].asfreq("D").interpolate(limit=3)
c1, c2 = st.columns(2)
lag = c1.slider("Lag (أيام)", 1, 30, 7, key="dt_lag")
win = c2.slider("نافذة Rolling", 3, 60, 14, key="dt_win")
view = pd.DataFrame({"sales": s, f"lag_{lag}": s.shift(lag), "lead_1": s.shift(-1),
                     f"rolling_mean_{win}": s.rolling(win).mean(),
                     f"rolling_centered_{win}": s.rolling(win, center=True).mean()})
st.dataframe(view.iloc[28:36].round(1))
seg = view.loc["2024-01-01":"2024-04-30"]
fig = go.Figure()
for i, c in enumerate(["sales", f"rolling_mean_{win}", f"rolling_centered_{win}"]):
    fig.add_trace(go.Scatter(x=seg.index, y=seg[c], name=c, line=dict(color=SEQUENCE[i], width=1 if i == 0 else 3)))
fig.update_layout(height=340, legend=dict(orientation="h", y=1.12))
plot(fig)
why("في مهام التنبؤ استخدم Lag وRolling العادي (المتأخر) فقط؛ لا Lead ولا Rolling مركزي.",
    "lead وcenter=True يستخدمان قيمًا مستقبلية غير متاحة لحظة التنبؤ ← Temporal leakage. "
    "لاحظ أن المتوسط المركزي «يسبق» الأحداث في الرسم.")

st.markdown("## الاتجاه والموسمية Trend & seasonality")
dec = ts.set_index("date")["sales"].asfreq("D").interpolate()
trend = dec.rolling(30, center=True).mean()
weekday = dec.groupby(dec.index.dayofweek).mean()
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dec.index, y=dec, name="sales", line=dict(color="#CCC", width=1)))
    fig.add_trace(go.Scatter(x=trend.index, y=trend, name="30-day trend", line=dict(color=PALETTE["coral"], width=3)))
    fig.add_vline(x=pd.Timestamp("2024-06-01"), line_dash="dash")  # annotations on date axes are added separately
    fig.add_annotation(x=pd.Timestamp("2024-06-01"), y=1, yref="paper", text="structural break?", showarrow=False, yanchor="bottom")
    fig.update_layout(height=320, legend=dict(orientation="h", y=1.12))
    plot(fig)
with c2:
    fig = go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=weekday,
                           marker_color=[PALETTE["purple"]] * 4 + [PALETTE["coral"]] * 2 + [PALETTE["purple"]]))
    fig.update_layout(title="متوسط المبيعات حسب اليوم (موسمية أسبوعية)", height=320,
                      yaxis=dict(range=[weekday.min() * 0.9, weekday.max() * 1.03]))
    plot(fig)
st.caption("المحور العمودي في الرسم الأيمن لا يبدأ من الصفر عمدًا لإظهار الفرق — ونصرّح بذلك (انظر وحدة التصوير).")


def _features(n_lags: int):
    d = ts.set_index("date")[["sales"]].asfreq("D")
    for k in range(1, n_lags + 1):
        d[f"lag_{k}"] = d["sales"].shift(k)
    d["roll7_mean"] = d["sales"].shift(1).rolling(7).mean()
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    return d.dropna().head(8).round(1)


code_lab("dt_feat", "هندسة خصائص زمنية آمنة",
         lambda p: ("d = df.set_index('date')[['sales']].asfreq('D')\n"
                    f"for k in range(1, {p['n_lags']} + 1):\n    d[f'lag_{{k}}'] = d['sales'].shift(k)\n"
                    "d['roll7_mean'] = d['sales'].shift(1).rolling(7).mean()   # shift first: past only\n"
                    "d['dow'] = d.index.dayofweek\nd['month'] = d.index.month"),
         _features, params=lambda: {"n_lags": st.slider("عدد الـLags", 1, 7, 3, key="dt_nl")},
         explanation="shift(1) قبل rolling يضمن أن متوسط الأيام السبعة لا يتضمن اليوم المُتنبأ به.")

comparison_table([
    {"الخاصية": "Calendar", "أمثلة": "month, dayofweek, holiday", "الخطر": "منخفض"},
    {"الخاصية": "Lag", "أمثلة": "sales_{t−1}, sales_{t−7}", "الخطر": "آمن إن كان متاحًا وقت التنبؤ"},
    {"الخاصية": "Rolling (trailing)", "أمثلة": "mean of last 7 days", "الخطر": "آمن مع shift"},
    {"الخاصية": "Rolling (centered) / Lead", "أمثلة": "mean of t−3..t+3", "الخطر": "Leakage"},
    {"الخاصية": "Time since event", "أمثلة": "days since last purchase", "الخطر": "يجب حسابه حتى لحظة القرار"},
])

if at_least("advanced"):
    st.markdown("## متقدم: التفكيك الموسمي")
    st.markdown("`statsmodels.tsa.seasonal.STL` يفكك السلسلة إلى Trend + Seasonal + Residual بشكل متين. "
                "البواقي الكبيرة بعد التفكيك مرشحة لتكون قيمًا شاذة زمنية (Contextual anomalies).")
if at_least("research"):
    researcher_note(["في بيانات Panel عرّف السنة المالية مقابل التقويمية بوضوح.",
                     "Structural breaks تُختبر إحصائيًا (Chow، Bai–Perron) قبل دمج الفترات في نموذج واحد."])
real_world(["ما المنطقة الزمنية للمصدر؟", "هل كل الأيام موجودة؟", "هل تغيرت طريقة التسجيل في تاريخ ما؟",
            "هل الخصائص الزمنية متاحة فعلًا لحظة التنبؤ؟"])

page_footer("datetime_data",
            takeaways=["حدد صيغة التاريخ صراحة.", "خزّن UTC مع المنطقة.", "اكشف الطوابع المفقودة بإعادة الفهرسة.",
                       "Lag وRolling المتأخر آمنان؛ Lead والمركزي تسرّب."],
            mistakes=["الاعتماد على الاستنتاج التلقائي للصيغة.", "sum بعد Resampling مع أيام مفقودة.", "Rolling مركزي في التنبؤ."])
