import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL    = "ws://localhost:8000/ws/metrics";
const API       = "http://localhost:8000";
const MAX_HIST  = 80;
const GPU_COLOR = "#FF6B35";
const NPU_COLOR = "#00E5A0";
const CPU_COLOR = "#7B7BFF";

// ─── Utility ─────────────────────────────────────────────────────
function sparkPath(data, key, W, H, lo, hi) {
  if (!data.length) return "";
  return "M " + data.map((d, i) => {
    const x = (i / (MAX_HIST - 1)) * W;
    const y = H - Math.max(0, Math.min(H, ((d[key] - lo) / (hi - lo)) * H));
    return `${x},${y}`;
  }).join(" L ");
}
function fmt(n, d = 1) { return n != null ? Number(n).toFixed(d) : "—"; }
function clr(proc) { return proc === "GPU" ? GPU_COLOR : proc === "NPU" ? NPU_COLOR : CPU_COLOR; }

// ─── Sub-components ──────────────────────────────────────────────
function Badge({ label, ok, warn }) {
  const col = ok ? NPU_COLOR : warn ? "#FFB347" : "#444";
  return (
    <div style={{
      display:"flex", alignItems:"center", gap:5,
      padding:"3px 10px", borderRadius:20, fontSize:10,
      background:`${col}11`, border:`1px solid ${col}44`, color:col,
    }}>
      <span style={{fontSize:9}}>{ok ? "●" : warn ? "◐" : "○"}</span>{label}
    </div>
  );
}

function StatCard({ label, value, unit, color, sub, flash }) {
  return (
    <div style={{
      flex:1, background:"rgba(255,255,255,0.025)",
      border:`1px solid rgba(255,255,255,0.06)`, borderRadius:10, padding:"14px 16px",
      animation: flash ? "cardFlash .5s ease" : "none",
    }}>
      <div style={{fontSize:9,color:"#555",letterSpacing:"0.15em",textTransform:"uppercase",marginBottom:6}}>{label}</div>
      <div style={{fontSize:28,fontFamily:"'Space Mono',monospace",color,lineHeight:1}}>
        {value}<span style={{fontSize:12,color:"#444",marginLeft:3}}>{unit}</span>
      </div>
      {sub && <div style={{fontSize:10,color:"#3a3a3a",marginTop:4,lineHeight:1.4}}>{sub}</div>}
    </div>
  );
}

function SparkChart({ history, dataKey, lo, hi, color, label, unit }) {
  const W=360, H=56;
  const path = sparkPath(history, dataKey, W, H, lo, hi);
  const last = history.length ? history[history.length-1][dataKey] : null;
  return (
    <div style={{flex:1, background:"rgba(0,0,0,0.18)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:8,padding:"10px 14px"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:6}}>
        <span style={{fontSize:9,color:"#555",letterSpacing:"0.12em",textTransform:"uppercase"}}>{label}</span>
        {last!=null&&<span style={{fontFamily:"'Space Mono',monospace",fontSize:14,color}}>{fmt(last)}<span style={{fontSize:10,color:"#444",marginLeft:2}}>{unit}</span></span>}
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{height:56,display:"block"}}>
        {[0.25,0.5,0.75].map(t=>(
          <line key={t} x1={0} y1={H*t} x2={W} y2={H*t} stroke="rgba(255,255,255,0.025)" strokeWidth={1}/>
        ))}
        {path&&<path d={`${path} L ${W},${H} L 0,${H} Z`} fill={`${color}14`}/>}
        {path&&<path d={path} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"/>}
      </svg>
    </div>
  );
}

function VideoFrame({ b64, processor, color }) {
  if (!b64) return (
    <div style={{width:"100%",aspectRatio:"16/9",background:"#0a0a0a",borderRadius:10,
      border:"1px solid rgba(255,255,255,0.04)",display:"flex",alignItems:"center",
      justifyContent:"center",color:"#2a2a2a",fontSize:12}}>
      Waiting for camera…
    </div>
  );
  return (
    <div style={{position:"relative",borderRadius:10,overflow:"hidden",border:`1px solid ${color}44`}}>
      <img src={`data:image/jpeg;base64,${b64}`} style={{width:"100%",display:"block"}} alt="AI inference"/>
      <div style={{
        position:"absolute",top:8,left:8,padding:"3px 10px",
        background:"rgba(0,0,0,0.75)",backdropFilter:"blur(6px)",
        border:`1px solid ${color}55`,borderRadius:20,
        fontSize:10,color,fontFamily:"'Space Mono',monospace",
      }}>
        {processor} · LIVE AI
      </div>
    </div>
  );
}

// ─── Sustainability Panel ─────────────────────────────────────────
function SustainPanel({ sustain, benchmark }) {
  const s = sustain;
  if (!s) return null;

  const pct = benchmark
    ? Math.round((1 - benchmark.npu?.wattage_mean / benchmark.gpu?.wattage_mean) * 100)
    : null;

  return (
    <div style={{
      background:"rgba(0,229,160,0.04)",border:"1px solid #00E5A022",
      borderRadius:12,padding:"16px 20px",
    }}>
      <div style={{
        fontSize:10,color:"#00E5A0",letterSpacing:"0.15em",
        textTransform:"uppercase",marginBottom:14,display:"flex",
        alignItems:"center",gap:8,
      }}>
        <span>🌱</span> Sustainability Tracker — Live Session
      </div>

      <div style={{display:"flex",gap:12,marginBottom:14}}>
        {/* CO2 saved */}
        <div style={{flex:1,background:"rgba(0,229,160,0.06)",borderRadius:8,padding:"12px 14px"}}>
          <div style={{fontSize:9,color:"#00A070",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:4}}>CO₂ Saved This Session</div>
          <div style={{fontSize:26,fontFamily:"'Space Mono',monospace",color:"#00E5A0",lineHeight:1}}>
            {fmt(s.co2_saved_session_g, 2)}<span style={{fontSize:11,color:"#006644",marginLeft:3}}>g</span>
          </div>
          <div style={{fontSize:10,color:"#004433",marginTop:4}}>
            ≈ {fmt(s.phone_charge_minutes, 0)} min phone charge equivalent
          </div>
        </div>

        {/* Energy saved */}
        <div style={{flex:1,background:"rgba(0,229,160,0.06)",borderRadius:8,padding:"12px 14px"}}>
          <div style={{fontSize:9,color:"#00A070",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:4}}>Energy Saved</div>
          <div style={{fontSize:26,fontFamily:"'Space Mono',monospace",color:"#00E5A0",lineHeight:1}}>
            {fmt(s.wh_saved_session, 1)}<span style={{fontSize:11,color:"#006644",marginLeft:3}}>mWh</span>
          </div>
          <div style={{fontSize:10,color:"#004433",marginTop:4}}>
            vs. always-on GPU baseline
          </div>
        </div>

        {/* Annual projection */}
        <div style={{flex:1,background:"rgba(0,229,160,0.06)",borderRadius:8,padding:"12px 14px"}}>
          <div style={{fontSize:9,color:"#00A070",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:4}}>Annual Projection</div>
          <div style={{fontSize:26,fontFamily:"'Space Mono',monospace",color:"#00E5A0",lineHeight:1}}>
            {fmt(s.co2_year_projection_kg, 2)}<span style={{fontSize:11,color:"#006644",marginLeft:3}}>kg CO₂</span>
          </div>
          <div style={{fontSize:10,color:"#004433",marginTop:4}}>
            per user · 4h unplugged/day · 250 days
          </div>
        </div>
      </div>

      {/* Benchmark comparison bar */}
      {benchmark && (
        <div>
          <div style={{fontSize:9,color:"#444",letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:8}}>
            Measured Power Comparison (from /benchmark run)
          </div>
          <div style={{marginBottom:6}}>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"#666",marginBottom:3}}>
              <span>GPU — {benchmark.gpu?.algorithm}</span>
              <span style={{fontFamily:"'Space Mono',monospace",color:GPU_COLOR}}>{benchmark.gpu?.wattage_mean}W avg</span>
            </div>
            <div style={{height:8,background:"rgba(255,255,255,0.04)",borderRadius:4,overflow:"hidden"}}>
              <div style={{width:"100%",height:"100%",background:`${GPU_COLOR}66`,borderRadius:4}}/>
            </div>
          </div>
          <div style={{marginBottom:12}}>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"#666",marginBottom:3}}>
              <span>NPU — {benchmark.npu?.algorithm}</span>
              <span style={{fontFamily:"'Space Mono',monospace",color:NPU_COLOR}}>{benchmark.npu?.wattage_mean}W avg</span>
            </div>
            <div style={{height:8,background:"rgba(255,255,255,0.04)",borderRadius:4,overflow:"hidden"}}>
              <div style={{
                width:`${100-pct}%`,height:"100%",
                background:`${NPU_COLOR}66`,borderRadius:4,
              }}/>
            </div>
          </div>
          <div style={{
            padding:"8px 12px",background:"rgba(0,229,160,0.06)",
            borderRadius:8,border:"1px solid #00E5A022",
            fontSize:11,color:"#00A070",lineHeight:1.7,
          }}>
            <strong style={{color:"#00E5A0"}}>{pct}% power reduction</strong>
            {" "}·{" "}
            {fmt(benchmark.savings?.wh_saved_per_8h_session, 1)} Wh saved per 8h session
            {" "}·{" "}
            {fmt(benchmark.savings?.co2_saved_per_session_g, 0)}g CO₂ per session
            {" "}·{" "}
            <strong style={{color:"#00E5A0"}}>
              {benchmark.savings?.co2_saved_per_year_kg} kg CO₂/year per user
            </strong>
            <br/>
            <span style={{fontSize:10,color:"#006644"}}>
              Grid factor: {benchmark.savings?.grid_factor_used}
              {" "}·{" "}
              1,000 laptops → {benchmark.savings?.enterprise_1000_laptops?.co2_tonnes_per_year} tonnes CO₂/year
            </span>
          </div>
        </div>
      )}
      {!benchmark && (
        <div style={{fontSize:11,color:"#1a3a2a",fontStyle:"italic"}}>
          Run <code style={{color:"#00A070"}}>POST /benchmark</code> to record real measured wattage data and unlock the comparison chart.
        </div>
      )}
    </div>
  );
}

// ─── Presentation Mode Panel ─────────────────────────────────────
function PresentationPanel({ pres, onStart, onStop }) {
  const [phaseSec, setPhaseSec] = useState(12);
  return (
    <div style={{
      background:"rgba(255,107,53,0.04)",border:"1px solid #FF6B3522",
      borderRadius:12,padding:"14px 18px",
    }}>
      <div style={{fontSize:10,color:GPU_COLOR,letterSpacing:"0.15em",textTransform:"uppercase",marginBottom:12,display:"flex",alignItems:"center",gap:8}}>
        <span>🎤</span> Presentation Mode
        {pres?.active && (
          <span style={{marginLeft:"auto",fontFamily:"'Space Mono',monospace",fontSize:12,color:pres.current_phase==="GPU"?GPU_COLOR:NPU_COLOR}}>
            {pres.current_phase} · {pres.phase_remaining}s
          </span>
        )}
      </div>

      {!pres?.active ? (
        <div style={{display:"flex",gap:10,alignItems:"center"}}>
          <div style={{fontSize:11,color:"#444",flex:1}}>
            Auto-switches GPU→NPU→GPU every N seconds. Never fumble the demo.
          </div>
          <div style={{display:"flex",alignItems:"center",gap:8}}>
            <input
              type="number" min={5} max={60} value={phaseSec}
              onChange={e=>setPhaseSec(Number(e.target.value))}
              style={{
                width:52,padding:"4px 8px",background:"rgba(255,255,255,0.05)",
                border:"1px solid rgba(255,255,255,0.1)",borderRadius:6,
                color:"#ccc",fontFamily:"'Space Mono',monospace",fontSize:13,textAlign:"center",
              }}
            />
            <span style={{fontSize:10,color:"#444"}}>sec/phase</span>
            <button onClick={()=>onStart(phaseSec)} style={{
              padding:"6px 16px",borderRadius:8,fontSize:12,cursor:"pointer",
              background:`${GPU_COLOR}22`,border:`1px solid ${GPU_COLOR}66`,
              color:GPU_COLOR,fontFamily:"'Syne',sans-serif",fontWeight:600,
            }}>Start Demo</button>
          </div>
        </div>
      ) : (
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          {/* Progress bar for current phase */}
          <div style={{flex:1}}>
            <div style={{fontSize:10,color:"#555",marginBottom:4}}>
              Next switch in <span style={{color:pres.current_phase==="GPU"?GPU_COLOR:NPU_COLOR,fontFamily:"'Space Mono',monospace"}}>{pres.phase_remaining}s</span>
            </div>
            <div style={{height:6,background:"rgba(255,255,255,0.05)",borderRadius:3,overflow:"hidden"}}>
              <div style={{
                width:`${(pres.phase_remaining/pres.phase_duration)*100}%`,
                height:"100%",
                background:pres.current_phase==="GPU"?GPU_COLOR:NPU_COLOR,
                borderRadius:3,transition:"width 1s linear",
              }}/>
            </div>
          </div>
          <button onClick={onStop} style={{
            padding:"6px 14px",borderRadius:8,fontSize:11,cursor:"pointer",
            background:"rgba(255,255,255,0.05)",border:"1px solid rgba(255,255,255,0.1)",
            color:"#666",fontFamily:"'Syne',sans-serif",
          }}>Stop</button>
        </div>
      )}
    </div>
  );
}

// ─── Model Rationale Card ─────────────────────────────────────────
function ModelRationale({ workload }) {
  if (!workload) return null;
  return (
    <div style={{
      padding:"10px 14px",background:"rgba(255,255,255,0.02)",
      border:"1px solid rgba(255,255,255,0.04)",borderRadius:8,
      fontSize:11,color:"#3a3a3a",lineHeight:1.65,
    }}>
      <span style={{color:"#666",fontWeight:700}}>Why this model? </span>
      {workload.algorithm_why}
    </div>
  );
}

// ─── Event Log ───────────────────────────────────────────────────
function EventLog({ events }) {
  const ref = useRef(null);
  useEffect(()=>{ if(ref.current) ref.current.scrollTop=ref.current.scrollHeight; },[events]);
  return (
    <div ref={ref} style={{
      height:110,overflowY:"auto",background:"rgba(0,0,0,0.25)",
      borderRadius:8,padding:"8px 12px",border:"1px solid rgba(255,255,255,0.04)",
      fontFamily:"'Space Mono',monospace",fontSize:10,
    }}>
      {!events.length&&<div style={{color:"#2a2a2a",marginTop:2}}>Waiting…</div>}
      {events.map((e,i)=>(
        <div key={i} style={{marginBottom:4,display:"flex",gap:8}}>
          <span style={{color:"#2a2a2a",flexShrink:0}}>{e.time}</span>
          <span style={{color:e.color}}>{e.msg}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────
export default function EcoScale() {
  const [data, setData]       = useState(null);
  const [history, setHistory] = useState([]);
  const [events, setEvents]   = useState([]);
  const [connected, setConn]  = useState(false);
  const [hw, setHw]           = useState(null);
  const [switchFlash, setFlash] = useState(false);
  const [benchRunning, setBenchRunning] = useState(false);
  const prevProc = useRef(null);
  const wsRef = useRef(null);

  const addEvent = useCallback((msg, color) => {
    const time = new Date().toLocaleTimeString("en-US",{hour12:false});
    setEvents(ev=>[...ev.slice(-50),{time,msg,color}]);
  }, []);

  const handleMsg = useCallback((raw) => {
    let pkt; try { pkt=JSON.parse(raw); } catch { return; }
    const w = pkt.workload;
    setData(pkt);
    setHistory(h=>[...h.slice(-(MAX_HIST-1)),{fps:w.fps,wattage:w.wattage,inf:w.inference_ms}]);
    if(pkt.hardware) setHw(pkt.hardware);

    if(prevProc.current && prevProc.current !== w.processor) {
      setFlash(true); setTimeout(()=>setFlash(false),600);
      const col = w.processor==="GPU" ? GPU_COLOR : NPU_COLOR;
      addEvent(`⚡ Workload shifted → ${w.profile_name} (${w.profile_label})`, col);
    }
    prevProc.current = w.processor;
  }, [addEvent]);

  useEffect(()=>{
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen  = ()=>{ setConn(true); addEvent("Connected to EcoScale backend","#7EF0B0"); };
      ws.onmessage = e => handleMsg(e.data);
      ws.onerror = ()=> addEvent("WebSocket error — backend running?","#FF6B6B");
      ws.onclose = ()=>{ setConn(false); setTimeout(connect,3000); };
    }
    connect();
    return ()=>wsRef.current?.close();
  },[handleMsg]);

  const forceProfile = async (mode) => {
    try {
      await fetch(`${API}/simulate/${mode}`,{method:"POST"});
      addEvent(`Manual override → ${mode.toUpperCase()}`, mode==="gpu"?GPU_COLOR:NPU_COLOR);
    } catch { addEvent("Backend unreachable","#FF6B6B"); }
  };

  const startBenchmark = async () => {
    setBenchRunning(true);
    addEvent("📊 Benchmark started — recording 30s per profile…","#FFB347");
    try {
      await fetch(`${API}/benchmark`,{method:"POST"});
      // poll until done
      const poll = setInterval(async()=>{
        const r = await fetch(`${API}/benchmark/results`);
        const j = await r.json();
        if(j.status !== "no_results") {
          clearInterval(poll);
          setBenchRunning(false);
          addEvent(`✓ Benchmark complete — GPU ${j.gpu?.wattage_mean}W / NPU ${j.npu?.wattage_mean}W`,"#00E5A0");
        }
      }, 5000);
    } catch { setBenchRunning(false); }
  };

  const startPres = async (sec) => {
    try {
      await fetch(`${API}/presentation/start?phase_seconds=${sec}`,{method:"POST"});
      addEvent(`🎤 Presentation mode started — ${sec}s per phase`,"#FF6B35");
    } catch {}
  };

  const stopPres = async () => {
    try {
      await fetch(`${API}/presentation/stop`,{method:"POST"});
      addEvent("Presentation mode stopped","#888");
    } catch {}
  };

  const w    = data?.workload;
  const p    = data?.power;
  const s    = data?.sustainability;
  const pres = data?.presentation;
  const bench= data?.benchmark;
  const isGPU = w?.processor === "GPU";
  const activeColor = clr(w?.processor);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        body{background:#07090C}
        ::-webkit-scrollbar{width:3px}
        ::-webkit-scrollbar-thumb{background:#1a1a1a;border-radius:2px}
        code{font-family:'Space Mono',monospace;font-size:0.9em}
        @keyframes scan{0%{transform:translateX(-100%)}100%{transform:translateX(200%)}}
        @keyframes pulse{0%{transform:scale(1);opacity:.7}100%{transform:scale(2.5);opacity:0}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        @keyframes flash{0%,100%{opacity:1}50%{opacity:0.25}}
        @keyframes cardFlash{0%,100%{background:rgba(255,255,255,0.025)}50%{background:rgba(0,229,160,0.08)}}
      `}</style>

      <div style={{
        minHeight:"100vh",background:"#07090C",color:"#DDD",
        fontFamily:"'Syne',sans-serif",padding:"24px 32px",
        animation:"fadeUp .5s ease both",
      }}>

        {/* ── Header ── */}
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:22}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <div style={{
              width:36,height:36,borderRadius:10,
              background:`linear-gradient(135deg,${GPU_COLOR},${NPU_COLOR})`,
              display:"flex",alignItems:"center",justifyContent:"center",fontSize:18,
            }}>⚡</div>
            <div>
              <h1 style={{fontSize:22,fontWeight:800,letterSpacing:"-0.02em",lineHeight:1}}>
                Eco<span style={{color:activeColor,transition:"color .5s"}}>Scale</span>
                <span style={{fontSize:11,color:"#333",fontWeight:400,marginLeft:10}}>v2</span>
              </h1>
              <div style={{fontSize:9,color:"#3a3a3a",letterSpacing:"0.14em",textTransform:"uppercase"}}>
                Context-Aware AI Workload Manager · HP Omen 16
              </div>
            </div>
          </div>

          <div style={{display:"flex",gap:7,flexWrap:"wrap",alignItems:"center"}}>
            <Badge label={`CUDA · ${hw?.gpu_name?.split(" ").slice(-2).join(" ") ?? "RTX 4060"}`} ok={hw?.cuda} warn={!hw?.cuda}/>
            <Badge label="XDNA NPU" ok={hw?.npu} warn={!hw?.npu}/>
            <Badge label="NVML Power" ok={hw?.nvml} warn={!hw?.nvml}/>
            <Badge label={connected?"Live":"Reconnecting"} ok={connected}/>
          </div>
        </div>

        {/* ── Power Banner ── */}
        <div style={{
          padding:"12px 20px",marginBottom:18,borderRadius:12,
          background:`linear-gradient(90deg,${activeColor}10,transparent)`,
          border:`1px solid ${activeColor}30`,
          display:"flex",alignItems:"center",gap:12,
          animation:switchFlash?"flash .4s ease":"none",
          transition:"border-color .5s,background .5s",
        }}>
          <div style={{position:"relative"}}>
            <div style={{width:8,height:8,borderRadius:"50%",background:activeColor}}/>
            <div style={{position:"absolute",top:0,left:0,width:8,height:8,borderRadius:"50%",
              background:activeColor,animation:"pulse 1.6s ease-out infinite"}}/>
          </div>
          <div style={{flex:1}}>
            <div style={{fontSize:13,fontWeight:700,color:activeColor}}>{w?.profile_label ?? "Initializing…"}</div>
            <div style={{fontSize:10,color:"#444",marginTop:1}}>
              {w?.profile_name} · {w?.algorithm?.split("—")[0]?.trim()} · {w?.resolution}
              {w?.inference_ms != null && ` · ${fmt(w.inference_ms,0)}ms inference`}
              {w?.detections != null && ` · ${w.detections} detection${w.detections!==1?"s":""}`}
            </div>
          </div>

          {/* Manual overrides */}
          {!pres?.active && (
            <div style={{display:"flex",gap:7}}>
              {["gpu","npu"].map(m=>(
                <button key={m} onClick={()=>forceProfile(m)} style={{
                  padding:"5px 13px",borderRadius:7,fontSize:11,cursor:"pointer",
                  fontFamily:"'Syne',sans-serif",fontWeight:600,
                  background:(m==="gpu"&&isGPU)||(m==="npu"&&!isGPU)?`${m==="gpu"?GPU_COLOR:NPU_COLOR}20`:"rgba(255,255,255,0.04)",
                  border:`1px solid ${(m==="gpu"&&isGPU)||(m==="npu"&&!isGPU)?(m==="gpu"?GPU_COLOR:NPU_COLOR)+"55":"rgba(255,255,255,0.07)"}`,
                  color:(m==="gpu"&&isGPU)||(m==="npu"&&!isGPU)?(m==="gpu"?GPU_COLOR:NPU_COLOR):"#3a3a3a",
                }}>
                  {m==="gpu"?"🔌 GPU":"🔋 NPU"}
                </button>
              ))}
            </div>
          )}

          {/* Battery */}
          <div style={{
            display:"flex",alignItems:"center",gap:6,padding:"4px 10px",
            background:"rgba(255,255,255,0.03)",borderRadius:8,border:"1px solid rgba(255,255,255,0.06)",
          }}>
            <span style={{fontSize:12}}>{p?.plugged?"🔌":"🔋"}</span>
            <span style={{fontFamily:"'Space Mono',monospace",fontSize:12,color:p?.plugged?GPU_COLOR:NPU_COLOR}}>
              {fmt(p?.battery_pct,0)}%
            </span>
          </div>
        </div>

        {/* ── Main Layout ── */}
        <div style={{display:"flex",gap:16,marginBottom:16}}>

          {/* Left — Video */}
          <div style={{flex:"0 0 440px",display:"flex",flexDirection:"column",gap:12}}>
            <VideoFrame b64={w?.frame_b64} processor={w?.processor} color={activeColor}/>
            <ModelRationale workload={w}/>

            {/* Stat cards */}
            <div style={{display:"flex",gap:10}}>
              <StatCard label="FPS"      value={fmt(w?.fps)}           unit="fps" color={activeColor} flash={switchFlash}/>
              <StatCard label="Latency"  value={fmt(w?.inference_ms,0)}unit="ms"  color={activeColor}/>
              <StatCard label="Power"    value={fmt(w?.wattage)}       unit="W"   color={activeColor} sub={hw?.nvml?"NVML real":"estimated"}/>
            </div>
          </div>

          {/* Right — Charts + controls */}
          <div style={{flex:1,display:"flex",flexDirection:"column",gap:12}}>

            <div style={{display:"flex",gap:10}}>
              <SparkChart history={history} dataKey="wattage" lo={0} hi={55} color={activeColor} label="Power Draw" unit="W"/>
              <SparkChart history={history} dataKey="fps"     lo={0} hi={70} color={activeColor} label="Frame Rate"  unit="fps"/>
              <SparkChart history={history} dataKey="inf"     lo={0} hi={200} color="#7B7BFF"   label="Inference"   unit="ms"/>
            </div>

            {/* Presentation mode */}
            <PresentationPanel pres={pres} onStart={startPres} onStop={stopPres}/>

            {/* Benchmark trigger */}
            <div style={{
              display:"flex",alignItems:"center",gap:14,padding:"12px 16px",
              background:"rgba(255,179,71,0.04)",border:"1px solid #FFB34722",
              borderRadius:10,
            }}>
              <div style={{flex:1,fontSize:11,color:"#555",lineHeight:1.6}}>
                <span style={{color:"#FFB347",fontWeight:700}}>📊 Benchmark</span>
                {" "} — runs both profiles for 30s each, records real wattage samples, and persists to{" "}
                <code>benchmark_results.json</code>. Unlocks the measured comparison chart below.
                {bench && (
                  <span style={{color:"#00A070"}}>{" "}Last run: {bench.recorded_at?.slice(0,16)?.replace("T"," ")}</span>
                )}
              </div>
              <button
                onClick={startBenchmark}
                disabled={benchRunning}
                style={{
                  padding:"7px 18px",borderRadius:8,fontSize:11,cursor:benchRunning?"not-allowed":"pointer",
                  background:benchRunning?"rgba(255,255,255,0.03)":"rgba(255,179,71,0.12)",
                  border:`1px solid ${benchRunning?"rgba(255,255,255,0.05)":"#FFB34755"}`,
                  color:benchRunning?"#333":"#FFB347",fontFamily:"'Syne',sans-serif",fontWeight:600,
                }}>
                {benchRunning?"Running…":"Run Benchmark"}
              </button>
            </div>

            {/* Event log */}
            <div>
              <div style={{fontSize:9,color:"#3a3a3a",letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:6}}>System Event Log</div>
              <EventLog events={events}/>
            </div>
          </div>
        </div>

        {/* ── Sustainability Panel ── */}
        <SustainPanel sustain={s} benchmark={bench}/>

        {/* ── Footer ── */}
        <div style={{
          marginTop:16,paddingTop:14,borderTop:"1px solid rgba(255,255,255,0.04)",
          display:"flex",justifyContent:"space-between",fontSize:9,color:"#1e1e1e",
        }}>
          <span>EcoScale v2 · HP Omen 16 · AMD Ryzen 7 + NVIDIA RTX 4060 + XDNA NPU</span>
          <span>AMD Slingshot Hackathon · Sustainable AI & Green Tech</span>
        </div>
      </div>
    </>
  );
}
