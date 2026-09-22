import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.callouts import real_world, researcher_note, why
from components.cards import comparison_table
from components.diagrams import mermaid
from core.page import page_footer, page_header
from core.state import at_least
from core.theme import PALETTE, SEQUENCE
from utils.datasets import customers_clean, daily_sales, transactions
from utils.plotting import plot

page_header("visualization")

st.markdown("## محدد الرسم · Chart Selector")
mermaid("""
flowchart TB
  Q[What do you want to show?] --> D[Distribution]
  Q --> C[Comparison]
  Q --> R[Relationship]
  Q --> T[Change over time]
  Q --> P[Part of a whole]
  D --> D1[Histogram / Density / ECDF / Boxplot / Violin]
  C --> C1[Bar sorted / Dot plot / Grouped bar]
  R --> R1[Scatter / Hexbin / Heatmap]
  T --> T1[Line / Area]
  P --> P1[Stacked bar 100% / Treemap<br/>Pie only for 2-3 parts]
""")
c1, c2, c3 = st.columns(3)
goal = c1.selectbox("الهدف", ["توزيع", "مقارنة", "علاقة", "تغير عبر الزمن", "جزء من كل"], key="viz_goal")
vtype = c2.selectbox("نوع المتغير/المتغيرات", ["رقمي واحد", "فئوي واحد", "رقمي × فئوي", "رقمي × رقمي", "فئوي × فئوي", "زمن × رقمي"],
                     key="viz_type")
size = c3.selectbox("حجم البيانات", ["صغير (< 1k)", "متوسط", "كبير (> 50k)"], key="viz_size")
RULES = {
    ("توزيع", "رقمي واحد"): ("Histogram أو ECDF", "Boxplot للمقارنة السريعة، Density للشكل الناعم"),
    ("توزيع", "رقمي × فئوي"): ("Boxplot / Violin حسب المجموعة", "Ridgeline عند مجموعات كثيرة"),
    ("مقارنة", "فئوي واحد"): ("Bar chart مرتب", "Dot plot عند فئات كثيرة"),
    ("مقارنة", "رقمي × فئوي"): ("Bar للمتوسط مع فترات ثقة، أو Boxplot", "لا تستخدم Bar للمتوسط دون إظهار التشتت"),
    ("علاقة", "رقمي × رقمي"): ("Scatter", "Hexbin أو كثافة ثنائية عند البيانات الكبيرة"),
    ("علاقة", "فئوي × فئوي"): ("Heatmap لجدول التوافق أو Stacked bar 100%", "Mosaic plot"),
    ("تغير عبر الزمن", "زمن × رقمي"): ("Line chart", "Area للمجاميع التراكمية"),
    ("جزء من كل", "فئوي واحد"): ("Stacked bar 100% أو Treemap", "Pie فقط لفئتين أو ثلاث"),
    ("جزء من كل", "فئوي × فئوي"): ("Stacked bar 100%", "Treemap هرمي"),
}
main, alt = RULES.get((goal, vtype), ("راجع نوع المتغيرات: هذا المزيج غير شائع لهذا الهدف", "ابدأ بالجدول"))
with st.container(border=True):
    st.markdown(f"**الرسم المقترح:** {main}  \n**بدائل:** {alt}")
    if size.startswith("كبير"):
        st.markdown("**مع البيانات الكبيرة:** شفافية، عينة عشوائية، Hexbin، أو تجميع مسبق — لتجنب Overplotting.")

st.markdown("## معرض الرسوم")
cust = customers_clean()
ts = daily_sales()
tr = transactions()
kind = st.segmented_control("الرسم", ["Histogram", "Bar", "Boxplot", "Violin", "Scatter", "Line", "Area", "Heatmap",
                                      "Density", "ECDF", "Treemap", "Map (concept)"], default="Bar", key="viz_kind")
if kind == "Histogram":
    fig = px.histogram(cust, x="monthly_spend", nbins=50, color_discrete_sequence=SEQUENCE)
elif kind == "Bar":
    s = cust["country"].value_counts().sort_values()
    fig = px.bar(x=s.values, y=s.index, orientation="h", color_discrete_sequence=SEQUENCE, labels={"x": "customers", "y": ""})
elif kind == "Boxplot":
    fig = px.box(cust, x="membership", y="monthly_spend", color="membership", color_discrete_sequence=SEQUENCE,
                 category_orders={"membership": ["Bronze", "Silver", "Gold", "Platinum"]})
elif kind == "Violin":
    fig = px.violin(cust, x="membership", y="age", box=True, color="membership", color_discrete_sequence=SEQUENCE,
                    category_orders={"membership": ["Bronze", "Silver", "Gold", "Platinum"]})
elif kind == "Scatter":
    fig = px.scatter(cust, x="annual_income", y="monthly_spend", color="membership", opacity=0.6,
                     color_discrete_sequence=SEQUENCE, log_x=True, log_y=True)
elif kind == "Line":
    fig = px.line(ts, x="date", y="sales", color_discrete_sequence=SEQUENCE)
elif kind == "Area":
    m = tr.assign(month=tr["datetime"].dt.to_period("M").dt.to_timestamp()).groupby(["month", "category"])["revenue"].sum().reset_index()
    fig = px.area(m, x="month", y="revenue", color="category", color_discrete_sequence=SEQUENCE)
elif kind == "Heatmap":
    m = tr.assign(dow=tr["datetime"].dt.day_name().str[:3], hour=tr["datetime"].dt.hour).pivot_table(
        index="dow", columns="hour", values="revenue", aggfunc="sum")
    m = m.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    fig = px.imshow(m, color_continuous_scale=["#FFFBEA", "#4DABF7", "#1864AB"], aspect="auto")
elif kind == "Density":
    fig = go.Figure()
    for i, g in enumerate(["Bronze", "Silver", "Gold", "Platinum"]):
        x = np.log10(cust.loc[cust["membership"] == g, "annual_income"])
        from scipy.stats import gaussian_kde
        grid = np.linspace(x.min() - 0.2, x.max() + 0.2, 200)
        fig.add_trace(go.Scatter(x=grid, y=gaussian_kde(x)(grid), name=g, fill="tozeroy", opacity=0.4,
                                 line=dict(color=SEQUENCE[i])))
    fig.update_layout(xaxis_title="log10(annual income)")
elif kind == "ECDF":
    fig = px.ecdf(cust, x="monthly_spend", color="membership", color_discrete_sequence=SEQUENCE)
elif kind == "Treemap":
    agg = tr.groupby(["category", "product"])["revenue"].sum().reset_index()
    fig = px.treemap(agg, path=["category", "product"], values="revenue", color="revenue",
                     color_continuous_scale=["#E7F5FF", "#4DABF7", "#1971C2"])
else:
    cities = pd.DataFrame({"city": ["Casablanca", "Cairo", "Riyadh", "Amman"], "lat": [33.57, 30.04, 24.71, 31.95],
                           "lon": [-7.59, 31.24, 46.68, 35.93]})
    rev = tr.groupby("store")["revenue"].sum()
    cities["revenue"] = cities["city"].map(rev)
    fig = px.scatter_geo(cities, lat="lat", lon="lon", size="revenue", hover_name="city", color_discrete_sequence=[PALETTE["coral"]],
                         projection="natural earth")
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#CED4DA")
fig.update_layout(height=430, title=kind)
plot(fig)

st.markdown("## الرسوم المضللة وكيف نصلحها")
mistake = st.selectbox("اختر خطأً شائعًا", ["Truncated axis", "Too many colors", "3D misuse", "Overplotting",
                                             "Dual axes / bad scales", "Chartjunk"], key="viz_mistake")
c1, c2 = st.columns(2)
if mistake == "Truncated axis":
    y = [102, 104]
    bad = go.Figure(go.Bar(x=["2023", "2024"], y=y, marker_color=[PALETTE["peach"], PALETTE["coral"]]))
    bad.update_layout(yaxis=dict(range=[101, 104.5]), title="❌ فرق 2% يبدو ×3")
    good = go.Figure(go.Bar(x=["2023", "2024"], y=y, marker_color=[PALETTE["peach"], PALETTE["coral"]], text=y, textposition="outside"))
    good.update_layout(yaxis=dict(range=[0, 115]), title="✅ الأعمدة تبدأ من الصفر")
    lesson = "أطوال الأعمدة تُقرأ كنسب؛ يجب أن تبدأ من الصفر. (الخطوط يمكنها ألا تبدأ من الصفر مع توضيح المحور.)"
elif mistake == "Too many colors":
    s = cust["country"].value_counts()
    bad = go.Figure(go.Pie(labels=s.index, values=s.values))
    bad.update_layout(title="❌ Pie بـ9 ألوان")
    good = go.Figure(go.Bar(x=s.values[::-1], y=s.index[::-1], orientation="h",
                            marker_color=[PALETTE["coral"] if c == "Morocco" else "#CED4DA" for c in s.index[::-1]]))
    good.update_layout(title="✅ Bar مرتب + لون واحد للإبراز")
    lesson = "استخدم اللون لإبراز ما يهم، لا لتلوين كل شيء. الأعمدة المرتبة أسهل مقارنة من الزوايا."
elif mistake == "3D misuse":
    vals = [45, 30, 25]
    bad = go.Figure(go.Pie(labels=["A", "B", "C"], values=vals, pull=[0.15, 0, 0], hole=0))
    bad.update_traces(marker=dict(line=dict(width=6, color="#999")))
    bad.update_layout(title="❌ تأثيرات «ثلاثية الأبعاد» تشوّه الزوايا (تقليد)")
    good = go.Figure(go.Bar(x=["A", "B", "C"], y=vals, marker_color=PALETTE["purple"], text=vals, textposition="outside"))
    good.update_layout(title="✅ ثنائي الأبعاد بقيم صريحة")
    lesson = "المنظور ثلاثي الأبعاد يجعل الشرائح القريبة تبدو أكبر. لا تضف بُعدًا لا يحمل بيانات."
elif mistake == "Overplotting":
    rng = np.random.default_rng(0)
    xx = rng.normal(0, 1, 30000)
    yy = xx * 0.5 + rng.normal(0, 1, 30000)
    bad = go.Figure(go.Scatter(x=xx, y=yy, mode="markers", marker=dict(color=PALETTE["purple"], size=6)))
    bad.update_layout(title="❌ 30,000 نقطة معتمة")
    good = go.Figure(go.Histogram2dContour(x=xx, y=yy, colorscale=[[0, "#FFFBEA"], [1, "#1971C2"]], ncontours=15))
    good.update_layout(title="✅ كثافة ثنائية")
    lesson = "مع النقاط الكثيرة: شفافية، عينة، Hexbin، أو كثافة ثنائية تكشف أين تتركز البيانات."
elif mistake == "Dual axes / bad scales":
    m = ts.set_index("date")[["sales", "temperature"]].resample("MS").mean()
    bad = go.Figure()
    bad.add_trace(go.Scatter(x=m.index, y=m["sales"], name="sales"))
    bad.add_trace(go.Scatter(x=m.index, y=m["temperature"], name="temperature", yaxis="y2"))
    bad.update_layout(yaxis2=dict(overlaying="y", side="right", range=[-40, 40]), title="❌ محوران باختيار مدى اعتباطي")
    z = (m - m.mean()) / m.std()
    good = go.Figure([go.Scatter(x=z.index, y=z[c], name=c) for c in z.columns])
    good.update_layout(title="✅ القيم موحّدة على محور واحد (أو رسمان منفصلان)")
    lesson = "المحوران المزدوجان يسمحان بصنع أي «علاقة» بتغيير المدى. افصل الرسوم أو وحّد المقاييس."
else:
    s = cust.groupby("membership")["monthly_spend"].mean().reindex(["Bronze", "Silver", "Gold", "Platinum"])
    bad = go.Figure(go.Bar(x=s.index, y=s.values, marker=dict(color=s.values, colorscale="Rainbow",
                                                             pattern=dict(shape="x"))))
    bad.update_layout(title="❌ Chartjunk: أنماط وألوان قوس قزح وشبكة كثيفة", plot_bgcolor="#EEE",
                      yaxis=dict(gridcolor="#888", dtick=10))
    good = go.Figure(go.Bar(x=s.index, y=s.values.round(1), marker_color=PALETTE["coral"], text=s.values.round(1),
                            textposition="outside"))
    good.update_layout(title="✅ حبر أقل، رسالة أوضح")
    lesson = "كل عنصر لا يحمل معلومة (Tufte: data-ink ratio) يشتت القارئ."
bad.update_layout(height=340)
good.update_layout(height=340)
with c1:
    plot(bad)
with c2:
    plot(good)
st.info(lesson, icon=":material/lightbulb:")

st.markdown("## مبادئ التصميم")
comparison_table([
    {"المبدأ": "عنوان يقول الرسالة", "مثال": "«الإنفاق يرتفع مع الدخل» بدل «Scatter plot»"},
    {"المبدأ": "محاور بوحدات", "مثال": "Monthly spend (MAD) لا «value»"},
    {"المبدأ": "ترتيب ذو معنى", "مثال": "الأعمدة تنازليًا، الفئات الترتيبية بترتيبها"},
    {"المبدأ": "عدم اليقين", "مثال": "فترات ثقة أو نطاقات، مع ذكر نوعها"},
    {"المبدأ": "إمكانية الوصول", "مثال": "لوحة آمنة لعمى الألوان، لا تعتمد على اللون وحده، تباين كافٍ"},
    {"المبدأ": "الرسوم التفاعلية", "مثال": "تمكّن الاستكشاف؛ لكن التقرير المطبوع يحتاج رسمًا ثابتًا واضحًا"},
])
why("اجعل الرسم يجيب عن سؤال واحد.", "الرسم الذي يحاول قول كل شيء لا يقول شيئًا؛ رسمان واضحان أفضل من رسم مزدحم.")
if at_least("research"):
    researcher_note(["في النشر العلمي: اذكر n، نوع أشرطة الخطأ (SD أم SE أم CI)، ومصدر البيانات.",
                     "تجنب «Bar + error bar» لإظهار التوزيعات؛ اعرض النقاط الفردية أو Violin (Weissgerber et al., 2015)."])
real_world(["من الجمهور؟ خبير أم إداري؟", "هل الرسالة واضحة في 5 ثوانٍ؟", "هل يُقرأ الرسم بالأبيض والأسود؟",
            "هل المحاور والمقاييس صادقة؟"])

page_footer("visualization",
            takeaways=["اختر الرسم حسب الهدف ونوع المتغيرات.", "الأعمدة تبدأ من الصفر.", "اللون للإبراز لا للزينة.",
                       "حل Overplotting بالشفافية أو الكثافة."],
            mistakes=["Pie بفئات كثيرة.", "محاور مقطوعة للأعمدة.", "3D بلا بيانات ثالثة.", "محوران بمدى اعتباطي."])
