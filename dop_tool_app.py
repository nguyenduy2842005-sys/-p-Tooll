
from __future__ import annotations
import io, json, math
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import beam_analysis_ui as ui
from dop_optimizer import (
    OptimizationSettings, analyze, force_envelopes, max_displacement,
    variational_frame_candidate, finite_design_optimize, kkt_verify_design
)
from dop_report import build_report

st.set_page_config(page_title="Đớp Tool — Structural Optimization", page_icon="🏗️", layout="wide")

def _state_df(key, columns):
    df = st.session_state.get(key)
    if df is None:
        return pd.DataFrame(columns=columns)
    return df.copy().reset_index(drop=True)

def _model_from_state():
    nd = _state_df("pf_nd_ed__data", ["x (m)", "y (m)"])
    el = _state_df("pf_el_ed__data", ["i","j","E","A","I","udl_local"])
    sp = _state_df("pf_sup_ed__data", ["node","Loại gối"])
    nl = _state_df("pf_nl_ed__data", ["node","Fx (kN)","Fy (kN)","Mz (kNm)"])

    nodes = [{"x":float(r["x (m)"]), "y":float(r["y (m)"])}
             for _,r in nd.iterrows() if pd.notna(r.get("x (m)")) and pd.notna(r.get("y (m)"))]
    elements = []
    for _,r in el.iterrows():
        if pd.isna(r.get("i")) or pd.isna(r.get("j")): continue
        elements.append({"i":int(r["i"]), "j":int(r["j"]),
                         "E":float(r.get("E",210e6)), "A":float(r.get("A",0.01)),
                         "I":float(r.get("I",1e-4)), "udl_local":float(r.get("udl_local",0) or 0)})
    supports = []
    for _,r in sp.iterrows():
        if pd.isna(r.get("node")): continue
        ux,uy,rz = ui._pf_label_to_bool(r.get("Loại gối","Ngàm"))
        supports.append({"node":int(r["node"]), "ux":ux, "uy":uy, "rz":rz})
    loads=[]
    for _,r in nl.iterrows():
        if pd.isna(r.get("node")): continue
        loads.append({"node":int(r["node"]), "Fx":float(r.get("Fx (kN)",0) or 0),
                       "Fy":float(r.get("Fy (kN)",0) or 0),
                       "Mz":float(r.get("Mz (kNm)",0) or 0)})
    return nodes,elements,supports,loads

def show_model_summary():
    nodes,elements,supports,loads = _model_from_state()
    r = st.session_state.get("pf_result")
    cols = st.columns(5)
    vals=[("Nút",len(nodes)),("Thanh",len(elements)),("Gối",len(supports)),
          ("Tải nút",len(loads)),("FEM","Đã chạy" if r is not None else "Chưa chạy")]
    for c,(a,b) in zip(cols,vals):
        c.metric(a,b)

def _section_svg(section_type, dims):
    """Draw a section in a self-contained SVG that cannot be clipped by the iframe."""
    # Large internal canvas + generous margins. Geometry is ALWAYS placed inside
    # [70,530] x [70,330], while the SVG itself is scaled with preserveAspectRatio.
    W, H = 600, 400
    GX0, GX1 = 75, 525
    GY0, GY1 = 72, 325
    GW, GH = GX1 - GX0, GY1 - GY0

    def rect(x, y, w, h, fill="#d9dde3", stroke="#111827", sw=3):
        return (f"<rect x='{x:.3f}' y='{y:.3f}' width='{max(w,0):.3f}' "
                f"height='{max(h,0):.3f}' fill='{fill}' stroke='{stroke}' "
                f"stroke-width='{sw}' vector-effect='non-scaling-stroke'/>" )

    parts=[]
    if section_type == "I/H":
        b,h,tf,tw=[float(dims[k]) for k in ("b","h","tf","tw")]
        # Protect the drawing even for unusual user input.
        tf=min(tf, h/2*0.95)
        tw=min(tw, b*0.95)
        sc=min(GW/max(b,1e-12), GH/max(h,1e-12))*0.82
        bw,bh,bt,wt=b*sc,h*sc,tf*sc,tw*sc
        x0=(GX0+GX1)/2-bw/2; y0=(GY0+GY1)/2-bh/2
        parts=[rect(x0,y0,bw,bt),
               rect(x0,y0+bh-bt,bw,bt),
               rect(x0+(bw-wt)/2,y0+bt,wt,max(bh-2*bt,0))]
        A=2*b*tf+(h-2*tf)*tw
        I=(b*h**3-(b-tw)*max(h-2*tf,0)**3)/12
        label=f"I/H: h={h*1000:.0f} mm · b={b*1000:.0f} mm · tw={tw*1000:.0f} mm · tf={tf*1000:.0f} mm"

    elif section_type == "Hộp chữ nhật rỗng":
        b,h,t=[float(dims[k]) for k in ("b","h","t")]
        t=min(t,b/2*0.95,h/2*0.95)
        sc=min(GW/max(b,1e-12),GH/max(h,1e-12))*0.82
        bw,bh,ti=b*sc,h*sc,t*sc
        x0=(GX0+GX1)/2-bw/2; y0=(GY0+GY1)/2-bh/2
        parts=[rect(x0,y0,bw,bh),
               rect(x0+ti,y0+ti,max(bw-2*ti,0),max(bh-2*ti,0),fill="#ffffff",stroke="#111827",sw=2)]
        A=b*h-max(b-2*t,0)*max(h-2*t,0)
        I=(b*h**3-max(b-2*t,0)*max(h-2*t,0)**3)/12
        label=f"Box: h={h*1000:.0f} mm · b={b*1000:.0f} mm · t={t*1000:.0f} mm"

    elif section_type == "Ống tròn":
        import math
        d,t=[float(dims[k]) for k in ("d","t")]
        t=min(t,d/2*0.95)
        outer_r=min(GW,GH)*0.34
        inner_r=outer_r*max(d-2*t,0)/max(d,1e-12)
        cx,cy=(GX0+GX1)/2,(GY0+GY1)/2
        parts=[f"<circle cx='{cx:.3f}' cy='{cy:.3f}' r='{outer_r:.3f}' fill='#d9dde3' stroke='#111827' stroke-width='3' vector-effect='non-scaling-stroke'/>",
               f"<circle cx='{cx:.3f}' cy='{cy:.3f}' r='{inner_r:.3f}' fill='#ffffff' stroke='#111827' stroke-width='2' vector-effect='non-scaling-stroke'/>"]
        A=math.pi*(d*d-(d-2*t)**2)/4
        I=math.pi*(d**4-(d-2*t)**4)/64
        label=f"Ống tròn: D={d*1000:.0f} mm · t={t*1000:.0f} mm"

    else:
        b,h=[float(dims[k]) for k in ("b","h")]
        sc=min(GW/max(b,1e-12),GH/max(h,1e-12))*0.82
        bw,bh=b*sc,h*sc
        x0=(GX0+GX1)/2-bw/2; y0=(GY0+GY1)/2-bh/2
        parts=[rect(x0,y0,bw,bh)]
        A=b*h; I=b*h**3/12
        label=f"Chữ nhật: b={b*1000:.0f} mm · h={h*1000:.0f} mm"

    svg=(
        "<style>html,body{margin:0!important;padding:0!important;width:100%;height:100%;overflow:hidden;background:transparent;}*{box-sizing:border-box}</style>"
        f"<svg xmlns='http://www.w3.org/2000/svg' width='100%' height='100%' "
        f"viewBox='0 0 {W} {H}' preserveAspectRatio='xMidYMid meet' "
        f"style='display:block;width:100%;height:100%;overflow:hidden;background:#fff;'>"
        f"<rect x='1' y='1' width='{W-2}' height='{H-2}' rx='12' fill='#fff' stroke='#c7ccd4' stroke-width='2'/>'"
        f"<text x='{W/2}' y='35' text-anchor='middle' font-family='Arial,sans-serif' font-size='16' font-weight='700' fill='#111827'>MẶT CẮT TIẾT DIỆN</text>"
        + ''.join(parts) +
        f"<text x='{W/2}' y='375' text-anchor='middle' font-family='Arial,sans-serif' font-size='12' fill='#374151'>{label} · A={A*1e4:.2f} cm² · I={I*1e8:.2f} cm⁴</text>"
        "</svg>"
    )
    # Remove an accidental quote from the frame string if present.
    svg=svg.replace("stroke-width='2'/>'", "stroke-width='2'/>")
    return svg,A,I

def render_section_panel():
    st.markdown("### 🧱 Mặt cắt tiết diện")
    st.caption("Hiển thị hình học tiết diện trực tiếp trong giao diện; A và I được tính từ kích thước thực.")
    c1,c2=st.columns([1,1.25])
    with c1:
        typ=st.selectbox("Loại tiết diện",["I/H","Hộp chữ nhật rỗng","Ống tròn","Chữ nhật đặc"],key="dop_sec_type")
        if typ=="I/H":
            b=st.number_input("b (mm)",50.0,1000.0,200.0,5.0,key="dop_sec_b")/1000; h=st.number_input("h (mm)",50.0,1500.0,300.0,5.0,key="dop_sec_h")/1000; tf=st.number_input("tf (mm)",2.0,100.0,10.0,1.0,key="dop_sec_tf")/1000; tw=st.number_input("tw (mm)",2.0,80.0,8.0,1.0,key="dop_sec_tw")/1000; dims={"b":b,"h":h,"tf":tf,"tw":tw}
        elif typ=="Hộp chữ nhật rỗng":
            b=st.number_input("b (mm)",50.0,1000.0,200.0,5.0,key="dop_box_b")/1000; h=st.number_input("h (mm)",50.0,1500.0,300.0,5.0,key="dop_box_h")/1000; t=st.number_input("t (mm)",2.0,80.0,8.0,1.0,key="dop_box_t")/1000; dims={"b":b,"h":h,"t":t}
        elif typ=="Ống tròn":
            d=st.number_input("D (mm)",20.0,1000.0,200.0,5.0,key="dop_pipe_d")/1000; t=st.number_input("t (mm)",1.0,80.0,8.0,1.0,key="dop_pipe_t")/1000; dims={"d":d,"t":t}
        else:
            b=st.number_input("b (mm)",20.0,1000.0,200.0,5.0,key="dop_rect_b")/1000; h=st.number_input("h (mm)",20.0,1500.0,300.0,5.0,key="dop_rect_h")/1000; dims={"b":b,"h":h}
    with c2:
        svg,A,I=_section_svg(typ,dims); components.html(svg, height=390, scrolling=False); st.write(f"**A = {A:.6g} m²** · **I = {I:.6g} m⁴**")


def render_optimizer():
    st.subheader("🧮 Tối ưu tiết diện bằng biến phân + FEM")
    nodes,elements,supports,loads = _model_from_state()
    if not elements:
        st.warning("Hãy sang tab **Mô hình FEM** để vẽ thanh, gán gối/tải và bấm Solve trước.")
        return

    r = st.session_state.get("pf_result")
    if r is None:
        st.info("Mô hình đã có nhưng chưa có kết quả FEM. Quay lại tab Mô hình FEM và bấm ▶ Solve.")
        return

    c1,c2,c3,c4 = st.columns(4)
    with c1: sigma = st.number_input("σ cho phép (MPa)", 1.0, 1000.0, 235.0, 1.0)
    with c2: dlim = st.number_input("Chuyển vị cho phép (mm)", 0.1, 500.0, 20.0, 0.5)
    with c3: Amin = st.number_input("Amin (mm²)", 1.0, 1e6, 100.0, 10.0)
    with c4: Amax = st.number_input("Amax (mm²)", 100.0, 1e7, 200000.0, 100.0)

    sw=st.checkbox("⚖️ Tính tự trọng của các thanh", value=True, key="dop_self_weight")
    settings=OptimizationSettings(
        sigma_allow=sigma*1000, disp_allow=dlim/1000,
        A_min=Amin/1e6, A_max=Amax/1e6,
        I_min=1e-8, I_max=5e-2,
        include_self_weight=sw,
    )
    st.session_state["dop_settings"] = settings
    st.session_state["dop_disp_allow"] = settings.disp_allow

    st.session_state["dop_include_self_weight"] = sw
    render_section_panel()

    st.markdown("### 1. Kết quả FEM đầu vào")
    show_model_summary()
    if sw:
        r_base = analyze(nodes,elements,supports,loads,settings,include_self_weight=True)
        st.info("FEM đầu vào cho tối ưu đã được chạy lại với **tự trọng phụ thuộc A hiện tại**.")
    else:
        r_base = r
    st.write(f"**Chuyển vị lớn nhất hiện tại:** {max_displacement(r_base)*1000:.4f} mm")

    env=force_envelopes(r_base)
    df=pd.DataFrame({
        "Thanh":[q["elem_idx"] for q in env],
        "Nmax (kN)":[q["Nmax"] for q in env],
        "Mmax (kNm)":[q["Mmax"] for q in env],
        "A hiện tại (m²)":[elements[q["elem_idx"]]["A"] for q in env],
        "I hiện tại (m⁴)":[elements[q["elem_idx"]]["I"] for q in env],
    })
    st.dataframe(df,use_container_width=True,hide_index=True)

    st.markdown("### 2. Ứng viên biến phân")
    cand,meta=variational_frame_candidate(r_base,elements,settings)
    cdf=pd.DataFrame({
        "Thanh":range(len(cand)),
        "A*(x) quy đổi (m²)":[x[0] for x in cand],
        "I*(x) quy đổi (m⁴)":[x[1] for x in cand],
        "λA":[x for x in meta["lambdas_A"]]
    })
    st.dataframe(cdf,use_container_width=True,hide_index=True)

    st.latex(r"A^*(x)=\max\left(\frac{|N(x)|}{\sigma_{allow}},\,|N(x)|\sqrt{\lambda/E}\right)")
    st.caption("Ứng viên liên tục được suy ra từ trường nội lực FEM; sau đó được dùng làm điểm khởi tạo cho bài toán tối ưu số có ràng buộc chuyển vị FEM trực tiếp.")

    if st.button("🚀 Tối ưu tiết diện",type="primary",use_container_width=True):
        with st.spinner("Đang tối ưu và phân tích lại FEM..."):
            out=finite_design_optimize(nodes,elements,supports,loads,settings)
        st.session_state["dop_opt"]=out
        st.session_state["dop_kkt"]=None
        st.rerun()

    out=st.session_state.get("dop_opt")
    if out:
        st.markdown("### 3. Kết quả sau tối ưu & hậu kiểm FEM")
        st.caption("Tiết diện sau tối ưu được đưa trở lại mô hình FEM để cập nhật tự trọng, nội lực và chuyển vị trước khi kết luận.")

        v0=float(out.get("objective_initial",0.0))
        v1=float(out.get("objective_final",0.0))
        d1=float(out.get("d_final",0.0))*1000.0
        dlim_mm=float(settings.disp_allow)*1000.0
        saving=(1.0-v1/v0)*100.0 if v0>0 else 0.0

        # Kết luận chính — ưu tiên cho người dùng, chi tiết kỹ thuật đưa xuống expander.
        if out.get("success"):
            st.success("✅ Tối ưu hội tụ. Tiết diện mới đã được **hậu kiểm bằng FEM**.")
        else:
            st.warning("⚠️ Bộ tối ưu chưa hội tụ hoàn toàn. Không nên xem kết quả là tiết diện cuối cùng cho đến khi kiểm tra lại các ràng buộc.")

        a,b,c,d=st.columns(4)
        a.metric("Thể tích thép trước",f"{v0:.4g} m³")
        b.metric("Thể tích thép sau",f"{v1:.4g} m³",delta=f"{saving:.1f}%",delta_color="normal")
        c.metric("Chuyển vị cuối",f"{d1:.4f} mm",delta=f"Giới hạn {dlim_mm:.1f} mm",delta_color="off")
        d.metric("Trạng thái FEM","Đã hậu kiểm" if out.get("opt_result") is not None else "Chưa hậu kiểm")

        # Thanh tiến trình trực quan cho chuyển vị.
        disp_ratio=d1/dlim_mm if dlim_mm>0 else 0.0
        disp_ratio=max(0.0,min(disp_ratio,1.0))
        st.markdown("**Mức sử dụng giới hạn chuyển vị**")
        st.progress(disp_ratio, text=f"{d1:.4f} / {dlim_mm:.1f} mm  ·  {disp_ratio*100:.1f}%")

        if out.get("self_weight_included"):
            st.info("⚖️ **Đã tính lại tự trọng:** A của tiết diện đã thay đổi → tự trọng thay đổi → FEM được chạy lại → N, M và chuyển vị cuối cùng được cập nhật.")
        else:
            st.info("ℹ️ Hậu kiểm FEM đã thực hiện, nhưng tùy chọn tự trọng đang tắt.")

        with st.expander("🔎 Xem chi tiết thuật toán và trạng thái tối ưu", expanded=False):
            st.write("**Thuật toán:** FEM → ứng viên biến phân → tối ưu số có ràng buộc → cập nhật A, I → FEM hậu kiểm → KKT.")
            st.write(f"**Trạng thái bộ tối ưu:** {out.get('message','Không có thông báo.')}")
            st.write(f"**Tiết kiệm thể tích thép:** {saving:.2f}%")
            st.write(f"**Chuyển vị hậu kiểm:** {d1:.4f} mm / {dlim_mm:.4f} mm")

        rdf=pd.DataFrame({
            "Thanh":range(len(out["A_final"])),
            "A trước (m²)":out["A_initial"],
            "A tối ưu (m²)":out["A_final"],
            "I trước (m⁴)":out["I_initial"],
            "I tối ưu (m⁴)":out["I_final"],
            "Chiều dài (m)":out["lengths"]
        })
        st.dataframe(rdf,use_container_width=True,hide_index=True)

        rr=out["opt_result"]
        env2=force_envelopes(rr)
        fig=go.Figure()
        for q in env2:
            i=q["elem_idx"]
            fig.add_trace(go.Scatter(x=q["x"],y=q["N"],mode="lines",name=f"E{i} N(x)"))
        fig.update_layout(title="Trường lực dọc sau tối ưu",xaxis_title="x cục bộ (m)",yaxis_title="N (kN)",height=360)
        st.plotly_chart(fig,use_container_width=True)

def render_kkt():
    st.subheader("✅ Kiểm tra KKT")
    out=st.session_state.get("dop_opt")
    if not out:
        st.info("Hãy chạy tối ưu trước.")
        return
    if st.button("🔎 Chạy kiểm tra KKT",type="primary"):
        nodes,elements,supports,loads=_model_from_state()
        settings=st.session_state.get("dop_settings", OptimizationSettings())
        kkt=kkt_verify_design(nodes,elements,supports,loads,settings,out)
        st.session_state["dop_kkt"]=kkt
    kkt=st.session_state.get("dop_kkt")
    if kkt:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Primal max violation",f"{kkt['primal_max_violation']:.3e}")
        c2.metric("Stationarity",f"{kkt['stationarity_norm']:.3e}")
        c3.metric("Complementarity",f"{kkt['complementarity_residual']:.3e}")
        c4.metric("Dual feasible","YES" if kkt["dual_feasible"] else "NO")
        st.latex(r"\nabla f(x^*)+J_g(x^*)^T\lambda=0,\qquad \lambda\ge0,\qquad g(x^*)\le0,\qquad \lambda_i g_i(x^*)=0")
        st.write("Các đại lượng trên là kiểm tra số tại nghiệm tối ưu. KKT không tự động chứng minh nghiệm toàn cục nếu bài toán không lồi.")

def render_report():
    st.subheader("📄 Thuyết minh")
    out=st.session_state.get("dop_opt")
    kkt=st.session_state.get("dop_kkt")
    if not out:
        st.info("Hãy hoàn thành FEM và tối ưu trước.")
        return
    if kkt is None:
        st.warning("Nên chạy kiểm tra KKT trước khi xuất thuyết minh.")
        return
    nodes,elements,supports,loads=_model_from_state()
    info={"n_nodes":len(nodes),"n_elements":len(elements),"n_supports":len(supports)}
    if st.button("📘 Tạo thuyết minh DOCX",type="primary",use_container_width=True):
        data=build_report(info,out,kkt)
        st.download_button("⬇️ Tải thuyết minh",data=data,file_name="DopTool_ThuyetMinh_ToiUu.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           use_container_width=True)

def main():
    st.markdown("""
    <style>
    .dop-title{font-size:34px;font-weight:850;letter-spacing:.5px}
    .dop-sub{opacity:.75;font-size:15px}
    </style>
    """,unsafe_allow_html=True)
    st.markdown('<div class="dop-title">🏗️ ĐỚP TOOL</div><div class="dop-sub">Structural Analysis → Variational Optimization → KKT Verification</div>',unsafe_allow_html=True)
    st.divider()

    tabs=st.tabs(["✏️ Mô hình FEM","🧮 Tối ưu biến phân","✅ KKT","📄 Thuyết minh"])
    with tabs[0]:
        ui.render_plane_frame()
    with tabs[1]:
        render_optimizer()
    with tabs[2]:
        render_kkt()
    with tabs[3]:
        render_report()

if __name__=="__main__":
    main()
