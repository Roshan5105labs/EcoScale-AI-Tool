import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL    = "ws://localhost:8000/ws/metrics";
const API       = "http://localhost:8000";
const MAX_HIST  = 80;
const GPU_COLOR = "#FF6B35";
const NPU_COLOR = "#00E5A0";
const AI_COLOR  = "#A78BFA";

function fmt(n, d=1) { return n != null ? Number(n).toFixed(d) : "—"; }
function clr(proc)   { return proc==="GPU"?GPU_COLOR:proc==="NPU"?NPU_COLOR:"#7B7BFF"; }

function sparkPath(data, key, W, H, lo, hi) {
  if (!data.length) return "";
  return "M " + data.map((d,i)=>{
    const x=(i/(MAX_HIST-1))*W;
    const y=H-Math.max(0,Math.min(H,((d[key]-lo)/(hi-lo))*H));
    return `${x},${y}`;
  }).join(" L ");
}

// ─── Signal Bar ───────────────────────────────────────────────────
function SignalBar({ label, value, max, unit, color, realValue, realUnit }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div style={{marginBottom:8}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3}}>
        <span style={{color:"#555",letterSpacing:"0.08em"}}>{label}</span>
        <div style={{display:"flex",gap:8}}>
          {realValue!=null&&<span style={{color:"#444",fontFamily:"'Space Mono',monospace"}}>{fmt(realValue,1)}{realUnit}</span>}
          <span style={{color,fontFamily:"'Space Mono',monospace",fontWeight:700}}>{fmt(pct,0)}%</span>
        </div>
      </div>
      <div style={{height:5,background:"rgba(255,255,255,0.04)",borderRadius:3,overflow:"hidden"}}>
        <div style={{
          width:`${pct}%`,height:"100%",
          background:`linear-gradient(90deg,${color}88,${color})`,
          borderRadius:3,transition:"width 0.5s ease",
        }}/>
      </div>
    </div>
  );
}

// ─── Policy Panel (STEP 4) ────────────────────────────────────────
function PolicyPanel({ policy, telemetry, policyStatus }) {
  if (!policy) return null;
  const isAI     = policy.mode === "ai_policy";
  const col      = policy.decision==="GPU"?GPU_COLOR:NPU_COLOR;
  const ss       = policy.signal_scores || {};
  const t        = telemetry || {};

  return (
    <div style={{
      background:"rgba(167,139,250,0.04)",
      border:"1px solid #A78BFA22",
      borderRadius:12,padding:"16px 18px",
    }}>
      {/* Header */}
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14}}>
        <span style={{fontSize:13}}>🧠</span>
        <span style={{fontSize:10,color:AI_COLOR,letterSpacing:"0.15em",textTransform:"uppercase",fontWeight:700}}>
          AI Policy Model
        </span>
        <span style={{
          marginLeft:4,fontSize:9,padding:"2px 8px",
          background:isAI?"rgba(167,139,250,0.15)":"rgba(255,255,255,0.05)",
          border:`1px solid ${isAI?"#A78BFA44":"rgba(255,255,255,0.1)"}`,
          borderRadius:20,color:isAI?AI_COLOR:"#444",
        }}>
          {isAI?"ACTIVE":"FALLBACK RULE"}
        </span>
        {policyStatus && (
          <span style={{marginLeft:"auto",fontSize:9,color:"#444"}}>
            {policyStatus.n_real_samples}r+{policyStatus.n_synthetic_samples}s samples
            · acc {policyStatus.accuracy?(policyStatus.accuracy*100).toFixed(1)+"%":"—"}
          </span>
        )}
      </div>

      <div style={{display:"flex",gap:16}}>
        {/* Decision output */}
        <div style={{
          flex:"0 0 130px",background:"rgba(0,0,0,0.2)",borderRadius:10,
          padding:"14px",border:`1px solid ${col}33`,
          display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",
        }}>
          <div style={{fontSize:10,color:"#444",marginBottom:6}}>DECISION</div>
          <div style={{fontSize:28,fontWeight:800,color:col,letterSpacing:"-0.02em"}}>
            {policy.decision}
          </div>
          <div style={{
            marginTop:6,fontSize:10,fontFamily:"'Space Mono',monospace",color:col,
          }}>
            {fmt(policy.confidence_pct,0)}% conf
          </div>
          {/* Confidence arc visualization */}
          <svg width={80} height={44} style={{marginTop:8}}>
            <path d="M 8 40 A 32 32 0 0 1 72 40" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={6}/>
            <path
              d={`M 8 40 A 32 32 0 0 1 72 40`}
              fill="none" stroke={col} strokeWidth={6}
              strokeDasharray={`${(policy.confidence_pct/100)*100} 100`}
              strokeLinecap="round"
              style={{transition:"stroke-dasharray 0.5s ease"}}
            />
            <text x={40} y={35} textAnchor="middle" fill={col} fontSize={10}
              fontFamily="'Space Mono',monospace">
              {fmt(policy.confidence_pct,0)}%
            </text>
          </svg>
          <div style={{fontSize:10,color:col==="GPU"?"#FF8C60":"#00C88A",marginTop:4,textAlign:"center"}}>
            {policy.reason}
          </div>
        </div>

        {/* Signal bars */}
        <div style={{flex:1}}>
          <div style={{fontSize:9,color:"#444",letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:10}}>
            Input Signals → Model
          </div>
          <SignalBar label="Battery %"   value={t.battery_pct??100}    max={100} unit="%" color={NPU_COLOR}
            realValue={t.battery_pct}  realUnit="%"/>
          <SignalBar label="Drain Rate"  value={Math.abs(t.drain_rate??0)*30} max={100} unit="" color="#FF6B6B"
            realValue={t.drain_rate}   realUnit="%/min"/>
          <SignalBar label="GPU Temp"    value={ss.gpu_temp??0}  max={100} unit="" color={GPU_COLOR}
            realValue={t.gpu_temp}     realUnit="°C"/>
          <SignalBar label="CPU Temp"    value={ss.cpu_temp??0}  max={100} unit="" color="#FFB347"
            realValue={t.cpu_temp}     realUnit="°C"/>
          <SignalBar label="Wattage"     value={ss.wattage??0}   max={100} unit="" color="#7B7BFF"
            realValue={t.wattage}      realUnit="W"/>
          <SignalBar label="FPS"         value={ss.fps??0}       max={100} unit="" color={NPU_COLOR}
            realValue={t.fps}          realUnit="fps"/>
        </div>

        {/* Probability bars */}
        <div style={{flex:"0 0 100px",display:"flex",flexDirection:"column",justifyContent:"center",gap:10}}>
          <div style={{fontSize:9,color:"#444",letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:4}}>
            Probability
          </div>
          {[["GPU",policy.prob_gpu,GPU_COLOR],["CPU",policy.prob_cpu,NPU_COLOR]].map(([lbl,prob,c])=>(
            <div key={lbl}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3}}>
                <span style={{color:c}}>{lbl}</span>
                <span style={{fontFamily:"'Space Mono',monospace",color:c}}>{fmt((prob??0)*100,0)}%</span>
              </div>
              <div style={{height:8,background:"rgba(255,255,255,0.04)",borderRadius:4,overflow:"hidden"}}>
                <div style={{
                  width:`${(prob??0)*100}%`,height:"100%",
                  background:c,borderRadius:4,transition:"width 0.4s ease",
                }}/>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── STEP 5 — Simulate Conditions Panel ──────────────────────────
function SimulatePanel({ simulating, onScenario, addEvent }) {
  const scenarios = [
    { key:"real",          label:"Real Data",       icon:"🔬", desc:"Live hardware readings",              color:"#555" },
    { key:"fast_drain",    label:"Fast Drain",       icon:"⚡", desc:"Plugged in but draining fast → CPU", color:NPU_COLOR },
    { key:"high_temp",     label:"High Temp",        icon:"🌡", desc:"GPU 88°C → thermal switch to CPU",   color:"#FF6B6B" },
    { key:"critical_batt", label:"Critical Battery", icon:"🔋", desc:"12% battery → minimum power",        color:"#FF4444" },
    { key:"night_mode",    label:"Night Mode",       icon:"🌙", desc:"2AM low activity → efficient",       color:AI_COLOR },
    { key:"peak_demand",   label:"Peak Demand",      icon:"🚀", desc:"FPS too low → GPU needed",           color:GPU_COLOR },
  ];

  return (
    <div style={{
      background:"rgba(255,107,53,0.04)",border:"1px solid #FF6B3520",
      borderRadius:12,padding:"14px 18px",
    }}>
      <div style={{fontSize:10,color:GPU_COLOR,letterSpacing:"0.15em",textTransform:"uppercase",marginBottom:12,display:"flex",alignItems:"center",gap:8}}>
        <span>🎮</span> Demo Scenarios
        {simulating && (
          <span style={{
            marginLeft:8,fontSize:9,padding:"2px 8px",
            background:"rgba(255,107,53,0.15)",border:"1px solid #FF6B3544",
            borderRadius:20,color:GPU_COLOR,
          }}>SIMULATION ACTIVE</span>
        )}
      </div>
      <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
        {scenarios.map(s=>(
          <button key={s.key} onClick={()=>onScenario(s.key)} style={{
            display:"flex",flexDirection:"column",gap:3,
            padding:"8px 12px",borderRadius:8,cursor:"pointer",
            fontFamily:"'Syne',sans-serif",textAlign:"left",
            background:s.key==="real"?"rgba(255,255,255,0.04)":`${s.color}12`,
            border:`1px solid ${s.key==="real"?"rgba(255,255,255,0.08)":s.color+"33"}`,
            color:s.key==="real"?"#555":s.color,
            transition:"all 0.2s",minWidth:100,
          }}>
            <span style={{fontSize:14}}>{s.icon}</span>
            <span style={{fontSize:11,fontWeight:700}}>{s.label}</span>
            <span style={{fontSize:9,opacity:0.7,lineHeight:1.3}}>{s.desc}</span>
          </button>
        ))}
      </div>
      <div style={{marginTop:10,fontSize:10,color:"#2a2a2a",lineHeight:1.6}}>
        Each scenario injects real signal values into the policy model.
        The AI makes its decision based on those signals — not the physical cable state.
        This proves the model is intelligent, not just detecting plug/unplug.
      </div>
    </div>
  );
}

// ─── Telemetry Proof Panel ────────────────────────────────────────
function TelemetryProof({ telemetry, simulating }) {
  if (!telemetry) return null;
  const t = telemetry;
  const fields = [
    ["Battery",      `${fmt(t.battery_pct,1)}%`,   t.battery_pct < 30 ? "#FF6B6B" : NPU_COLOR],
    ["Drain Rate",   `${fmt(t.drain_rate,2)}%/min`, t.drain_rate < -1.5 ? "#FF6B6B" : "#666"],
    ["GPU Temp",     `${fmt(t.gpu_temp,1)}°C`,      t.gpu_temp > 80 ? "#FF6B6B" : GPU_COLOR],
    ["CPU Temp",     `${fmt(t.cpu_temp,1)}°C`,      t.cpu_temp > 75 ? "#FFB347" : "#666"],
    ["Wattage",      `${fmt(t.wattage,1)}W`,        GPU_COLOR],
    ["FPS",          `${fmt(t.fps,1)}`,             NPU_COLOR],
    ["Inf. Latency", `${fmt(t.inference_ms,0)}ms`,  "#7B7BFF"],
    ["CPU Usage",    `${fmt(t.cpu_usage,1)}%`,      "#888"],
    ["Plugged",      t.plugged ? "Yes" : "No",      t.plugged ? GPU_COLOR : NPU_COLOR],
    ["Hour",         `${t.hour_of_day}:00`,         "#555"],
  ];

  return (
    <div style={{
      background:"rgba(0,0,0,0.15)",border:"1px solid rgba(255,255,255,0.04)",
      borderRadius:10,padding:"12px 16px",
    }}>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
        <span style={{fontSize:9,color:"#444",letterSpacing:"0.12em",textTransform:"uppercase"}}>
          Live Hardware Signals
        </span>
        {simulating && (
          <span style={{fontSize:9,color:"#FF6B35",padding:"1px 6px",
            background:"rgba(255,107,53,0.1)",border:"1px solid #FF6B3533",borderRadius:10}}>
            SIM
          </span>
        )}
        <span style={{marginLeft:"auto",fontSize:9,color:"#2a2a2a"}}>
          Real psutil + pynvml + WMI readings
        </span>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:8}}>
        {fields.map(([label,value,color])=>(
          <div key={label} style={{
            background:"rgba(255,255,255,0.02)",borderRadius:6,padding:"8px 10px",
            border:"1px solid rgba(255,255,255,0.04)",
          }}>
            <div style={{fontSize:8,color:"#444",letterSpacing:"0.1em",marginBottom:3}}>{label}</div>
            <div style={{fontSize:13,fontFamily:"'Space Mono',monospace",color}}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Model Training Panel ─────────────────────────────────────────
function TrainingPanel({ policyStatus, telemetryRows, onCollect, onTrain, collecting, training }) {
  const rows = telemetryRows || 0;
  const trained = policyStatus?.is_trained;
  const acc = policyStatus?.accuracy;

  return (
    <div style={{
      background:"rgba(167,139,250,0.04)",border:"1px solid #A78BFA18",
      borderRadius:10,padding:"12px 16px",
    }}>
      <div style={{fontSize:10,color:AI_COLOR,letterSpacing:"0.15em",textTransform:"uppercase",marginBottom:12}}>
        🧪 Policy Model Training
      </div>
      <div style={{display:"flex",gap:10,alignItems:"center"}}>
        <div style={{flex:1,fontSize:11,color:"#444",lineHeight:1.7}}>
          <span style={{color:rows>50?NPU_COLOR:"#555"}}>
            {rows} real telemetry rows
          </span>
          {" "}collected from your hardware.
          {trained && <span style={{color:AI_COLOR}}> Model trained — accuracy {fmt(acc*100,1)}%.</span>}
          {!trained && <span style={{color:"#444"}}> Train model to activate AI decisions.</span>}
        </div>

        <div style={{display:"flex",gap:8}}>
          <button onClick={onCollect} disabled={collecting} style={{
            padding:"7px 14px",borderRadius:8,fontSize:11,cursor:collecting?"not-allowed":"pointer",
            fontFamily:"'Syne',sans-serif",fontWeight:600,
            background:collecting?"rgba(255,255,255,0.03)":"rgba(0,229,160,0.1)",
            border:`1px solid ${collecting?"rgba(255,255,255,0.05)":"#00E5A044"}`,
            color:collecting?"#333":NPU_COLOR,
          }}>
            {collecting?"Collecting…":"Collect Data (2 min)"}
          </button>
          <button onClick={onTrain} disabled={training||rows<10} style={{
            padding:"7px 14px",borderRadius:8,fontSize:11,
            cursor:(training||rows<10)?"not-allowed":"pointer",
            fontFamily:"'Syne',sans-serif",fontWeight:600,
            background:(training||rows<10)?"rgba(255,255,255,0.03)":"rgba(167,139,250,0.12)",
            border:`1px solid ${(training||rows<10)?"rgba(255,255,255,0.05)":"#A78BFA44"}`,
            color:(training||rows<10)?"#333":AI_COLOR,
          }}>
            {training?"Training…":"Train Model"}
          </button>
        </div>
      </div>

      {/* Feature importances */}
      {policyStatus?.feature_importances && Object.keys(policyStatus.feature_importances).length > 0 && (
        <div style={{marginTop:12}}>
          <div style={{fontSize:9,color:"#444",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:6}}>
            Feature Importances — what the model learned matters most
          </div>
          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
            {Object.entries(policyStatus.feature_importances).slice(0,6).map(([k,v])=>(
              <div key={k} style={{
                display:"flex",flexDirection:"column",gap:2,
                padding:"6px 8px",background:"rgba(167,139,250,0.06)",
                borderRadius:6,border:"1px solid #A78BFA22",minWidth:80,
              }}>
                <div style={{fontSize:8,color:"#555"}}>{k.replace("_"," ")}</div>
                <div style={{height:3,background:"rgba(255,255,255,0.05)",borderRadius:2}}>
                  <div style={{width:`${v*100}%`,height:"100%",background:AI_COLOR,borderRadius:2}}/>
                </div>
                <div style={{fontSize:9,fontFamily:"'Space Mono',monospace",color:AI_COLOR}}>
                  {fmt(v*100,1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Sustainability Panel ─────────────────────────────────────────
function SustainPanel({ sustain, benchmark }) {
  const s=sustain, b=benchmark;
  if(!s) return null;
  const pct=b?Math.round((1-b.npu?.wattage_mean/b.gpu?.wattage_mean)*100):null;

  return (
    <div style={{background:"rgba(0,229,160,0.04)",border:"1px solid #00E5A020",borderRadius:12,padding:"16px 20px"}}>
      <div style={{fontSize:10,color:NPU_COLOR,letterSpacing:"0.15em",textTransform:"uppercase",marginBottom:14,display:"flex",alignItems:"center",gap:8}}>
        🌱 Sustainability Tracker — Live Session
      </div>
      <div style={{display:"flex",gap:12,marginBottom:b?14:0}}>
        {[
          ["CO₂ Saved", fmt(s.co2_saved_session_g,2), "g", `≈${fmt(s.phone_charge_minutes,0)}min phone charge`],
          ["Energy Saved", fmt(s.wh_saved_session,1), "mWh", "vs GPU baseline"],
          ["Annual Projection", fmt(s.co2_year_projection_kg,2), "kg CO₂", "4h unplugged · 250 days/yr"],
        ].map(([label,val,unit,sub])=>(
          <div key={label} style={{flex:1,background:"rgba(0,229,160,0.06)",borderRadius:8,padding:"12px 14px",border:"1px solid #00E5A015"}}>
            <div style={{fontSize:9,color:"#00A070",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:4}}>{label}</div>
            <div style={{fontSize:26,fontFamily:"'Space Mono',monospace",color:NPU_COLOR,lineHeight:1}}>
              {val}<span style={{fontSize:11,color:"#006644",marginLeft:3}}>{unit}</span>
            </div>
            <div style={{fontSize:10,color:"#004433",marginTop:4}}>{sub}</div>
          </div>
        ))}
      </div>
      {b && (
        <div>
          {[["GPU",b.gpu?.algorithm,b.gpu?.wattage_mean,GPU_COLOR,100],
            ["CPU/NPU",b.npu?.algorithm,b.npu?.wattage_mean,NPU_COLOR,100-pct]].map(([lbl,alg,w,c,wp])=>(
            <div key={lbl} style={{marginBottom:8}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"#555",marginBottom:3}}>
                <span>{lbl} — {alg}</span>
                <span style={{fontFamily:"'Space Mono',monospace",color:c}}>{fmt(w,2)}W avg</span>
              </div>
              <div style={{height:7,background:"rgba(255,255,255,0.04)",borderRadius:4,overflow:"hidden"}}>
                <div style={{width:`${wp}%`,height:"100%",background:`${c}66`,borderRadius:4}}/>
              </div>
            </div>
          ))}
          <div style={{
            marginTop:10,padding:"8px 12px",background:"rgba(0,229,160,0.06)",
            borderRadius:8,border:"1px solid #00E5A020",fontSize:11,color:"#00A070",lineHeight:1.8,
          }}>
            <strong style={{color:NPU_COLOR}}>{pct}% power reduction</strong>
            {" "}·{" "}{fmt(b.savings?.wh_saved_per_8h_session,1)} Wh saved per 8h session
            {" "}·{" "}<strong style={{color:NPU_COLOR}}>{b.savings?.co2_saved_per_year_kg} kg CO₂/user/year</strong>
            <br/>
            <span style={{fontSize:9,color:"#006644"}}>
              {b.savings?.grid_factor}
              {" "}·{" "}1,000 laptops → {b.savings?.enterprise_1000_laptops?.co2_tonnes_per_year} tonnes CO₂/year
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Sparkline ───────────────────────────────────────────────────
function SparkChart({ history, dataKey, lo, hi, color, label, unit }) {
  const W=340,H=50;
  const path=sparkPath(history,dataKey,W,H,lo,hi);
  const last=history.length?history[history.length-1][dataKey]:null;
  return (
    <div style={{flex:1,background:"rgba(0,0,0,0.18)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:8,padding:"10px 12px"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:4}}>
        <span style={{fontSize:9,color:"#555",letterSpacing:"0.1em",textTransform:"uppercase"}}>{label}</span>
        {last!=null&&<span style={{fontFamily:"'Space Mono',monospace",fontSize:13,color}}>{fmt(last)}<span style={{fontSize:9,color:"#444",marginLeft:2}}>{unit}</span></span>}
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{height:50,display:"block"}}>
        {[0.3,0.6].map(t=><line key={t} x1={0} y1={H*t} x2={W} y2={H*t} stroke="rgba(255,255,255,0.025)" strokeWidth={1}/>)}
        {path&&<path d={`${path} L ${W},${H} L 0,${H} Z`} fill={`${color}14`}/>}
        {path&&<path d={path} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"/>}
      </svg>
    </div>
  );
}

function EventLog({ events }) {
  const ref=useRef(null);
  useEffect(()=>{if(ref.current)ref.current.scrollTop=ref.current.scrollHeight;},[events]);
  return (
    <div ref={ref} style={{
      height:100,overflowY:"auto",background:"rgba(0,0,0,0.25)",
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
  const [data,setData]           = useState(null);
  const [history,setHistory]     = useState([]);
  const [events,setEvents]       = useState([]);
  const [connected,setConn]      = useState(false);
  const [hw,setHw]               = useState(null);
  const [policyStatus,setPStat]  = useState(null);
  const [telemetryRows,setTRows] = useState(0);
  const [switchFlash,setFlash]   = useState(false);
  const [collecting,setCollect]  = useState(false);
  const [training,setTraining]   = useState(false);
  const prevProc=useRef(null);
  const wsRef=useRef(null);

  const addEvent=useCallback((msg,color)=>{
    const time=new Date().toLocaleTimeString("en-US",{hour12:false});
    setEvents(ev=>[...ev.slice(-50),{time,msg,color}]);
  },[]);

  const handleMsg=useCallback((raw)=>{
    let pkt;try{pkt=JSON.parse(raw);}catch{return;}
    const w=pkt.workload;
    setData(pkt);
    setHistory(h=>[...h.slice(-(MAX_HIST-1)),{fps:w.fps,wattage:w.wattage,inf:w.inference_ms}]);
    if(pkt.hardware) setHw(pkt.hardware);

    if(prevProc.current&&prevProc.current!==w.processor){
      setFlash(true);setTimeout(()=>setFlash(false),600);
      const pol=pkt.policy;
      const col=w.processor==="GPU"?GPU_COLOR:NPU_COLOR;
      const reason=pol?.reason?"— "+pol.reason:"";
      addEvent(`⚡ AI shifted → ${w.profile_name} (${fmt(pol?.confidence_pct,0)}% conf) ${reason}`,col);
    }
    prevProc.current=w.processor;
  },[addEvent]);

  useEffect(()=>{
    function connect(){
      const ws=new WebSocket(WS_URL);wsRef.current=ws;
      ws.onopen=()=>{setConn(true);addEvent("Connected to EcoScale v3","#7EF0B0");};
      ws.onmessage=e=>handleMsg(e.data);
      ws.onerror=()=>addEvent("WS error — is backend running?","#FF6B6B");
      ws.onclose=()=>{setConn(false);setTimeout(connect,3000);};
    }
    connect();
    return()=>wsRef.current?.close();
  },[handleMsg]);

  // Poll policy status separately
  useEffect(()=>{
    const poll=setInterval(async()=>{
      try{
        const r=await fetch(`${API}/policy/status`);
        const j=await r.json();
        setPStat(j);
      }catch{}
      try{
        const r=await fetch(`${API}/policy/telemetry`);
        const j=await r.json();
        setTRows(j.total||0);
      }catch{}
    },5000);
    return()=>clearInterval(poll);
  },[]);

  const handleScenario=async(scenario)=>{
    try{
      await fetch(`${API}/simulate/conditions?scenario=${scenario}`,{method:"POST"});
      const icons={"real":"🔬","fast_drain":"⚡","high_temp":"🌡",
                   "critical_batt":"🔋","night_mode":"🌙","peak_demand":"🚀"};
      addEvent(`${icons[scenario]||"🎮"} Scenario: ${scenario}`,GPU_COLOR);
    }catch{addEvent("Backend unreachable","#FF6B6B");}
  };

  const handleCollect=async()=>{
    setCollect(true);
    addEvent("🔬 Collecting real telemetry — 2 min per profile…",AI_COLOR);
    try{
      await fetch(`${API}/policy/collect?seconds=120`,{method:"POST"});
      setTimeout(()=>setCollect(false),250000);
    }catch{setCollect(false);}
  };

  const handleTrain=async()=>{
    setTraining(true);
    addEvent("🧠 Training policy model on real data…",AI_COLOR);
    try{
      const r=await fetch(`${API}/policy/train`,{method:"POST"});
      const j=await r.json();
      setTraining(false);
      if(j.accuracy){
        addEvent(`✓ Model trained — ${j.accuracy} accuracy · ${j.n_real_samples} real samples`,NPU_COLOR);
        setPStat(prev=>({...prev,...j,is_trained:true}));
      }
    }catch{setTraining(false);}
  };

  const forceProfile=async(mode)=>{
    try{
      await fetch(`${API}/simulate/${mode}`,{method:"POST"});
      addEvent(`Manual override → ${mode.toUpperCase()} (policy paused)`,mode==="gpu"?GPU_COLOR:NPU_COLOR);
    }catch{}
  };

  const resumePolicy=async()=>{
    try{
      await fetch(`${API}/policy/resume`,{method:"POST"});
      addEvent("Policy model resumed — AI in control",AI_COLOR);
    }catch{}
  };

  const w=data?.workload, p=data?.power, s=data?.sustainability;
  const pol=data?.policy, t=data?.telemetry, bench=data?.benchmark;
  const pres=data?.presentation, simulating=data?.simulating;
  const isGPU=w?.processor==="GPU";
  const activeColor=clr(w?.processor);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        body{background:#07090C}
        ::-webkit-scrollbar{width:3px}
        ::-webkit-scrollbar-thumb{background:#1a1a1a;border-radius:2px}
        code{font-family:'Space Mono',monospace}
        @keyframes pulse{0%{transform:scale(1);opacity:.7}100%{transform:scale(2.5);opacity:0}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        @keyframes flash{0%,100%{opacity:1}50%{opacity:0.2}}
      `}</style>

      <div style={{
        minHeight:"100vh",background:"#07090C",color:"#DDD",
        fontFamily:"'Syne',sans-serif",padding:"22px 28px",
        animation:"fadeUp .5s ease both",
      }}>

        {/* Header */}
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:18}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <div style={{width:34,height:34,borderRadius:10,
              background:`linear-gradient(135deg,${GPU_COLOR},${NPU_COLOR})`,
              display:"flex",alignItems:"center",justifyContent:"center",fontSize:17}}>⚡</div>
            <div>
              <h1 style={{fontSize:20,fontWeight:800,letterSpacing:"-0.02em",lineHeight:1}}>
                Eco<span style={{color:activeColor,transition:"color .5s"}}>Scale</span>
                <span style={{fontSize:10,color:"#333",fontWeight:400,marginLeft:8}}>v3 · AI Policy Edition</span>
              </h1>
              <div style={{fontSize:9,color:"#2a2a2a",letterSpacing:"0.14em",textTransform:"uppercase"}}>
                Context-Aware AI Workload Manager · HP Omen 16 · AMD Slingshot
              </div>
            </div>
          </div>
          <div style={{display:"flex",gap:6,flexWrap:"wrap",alignItems:"center"}}>
            {[
              [`CUDA·RTX 4060`,hw?.cuda],
              [`XDNA NPU`,hw?.npu],
              [`NVML Power`,hw?.nvml],
              [`WMI Temp`,hw?.wmi],
              [`AI Policy`,policyStatus?.is_trained],
              [connected?"Live":"Reconnecting",connected],
            ].map(([label,ok])=>(
              <div key={label} style={{
                display:"flex",alignItems:"center",gap:4,padding:"3px 9px",borderRadius:20,
                fontSize:9,background:`${ok?NPU_COLOR:GPU_COLOR}10`,
                border:`1px solid ${ok?NPU_COLOR:GPU_COLOR}33`,color:ok?NPU_COLOR:"#555",
              }}>
                <span style={{fontSize:8}}>{ok?"●":"○"}</span>{label}
              </div>
            ))}
          </div>
        </div>

        {/* Power Banner */}
        <div style={{
          padding:"12px 18px",marginBottom:16,borderRadius:12,
          background:`linear-gradient(90deg,${activeColor}10,transparent)`,
          border:`1px solid ${activeColor}28`,
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
            <div style={{fontSize:13,fontWeight:700,color:activeColor}}>{w?.profile_label??"Initializing…"}</div>
            <div style={{fontSize:10,color:"#3a3a3a",marginTop:1}}>
              {w?.profile_name} · {w?.algorithm} · {w?.resolution}
              {w?.inference_ms!=null&&` · ${fmt(w.inference_ms,0)}ms`}
              {w?.detections!=null&&` · ${w.detections} objects`}
              {pol?.mode==="ai_policy"&&<span style={{color:AI_COLOR}}> · 🧠 AI decided</span>}
            </div>
          </div>
          <div style={{display:"flex",gap:6}}>
            <button onClick={()=>forceProfile("gpu")} style={{
              padding:"5px 12px",borderRadius:7,fontSize:10,cursor:"pointer",
              fontFamily:"'Syne',sans-serif",fontWeight:600,
              background:isGPU?`${GPU_COLOR}20`:"rgba(255,255,255,0.04)",
              border:`1px solid ${isGPU?GPU_COLOR+"55":"rgba(255,255,255,0.07)"}`,
              color:isGPU?GPU_COLOR:"#3a3a3a",
            }}>🔌 GPU</button>
            <button onClick={()=>forceProfile("npu")} style={{
              padding:"5px 12px",borderRadius:7,fontSize:10,cursor:"pointer",
              fontFamily:"'Syne',sans-serif",fontWeight:600,
              background:!isGPU?`${NPU_COLOR}20`:"rgba(255,255,255,0.04)",
              border:`1px solid ${!isGPU?NPU_COLOR+"55":"rgba(255,255,255,0.07)"}`,
              color:!isGPU?NPU_COLOR:"#3a3a3a",
            }}>🔋 CPU</button>
            <button onClick={resumePolicy} style={{
              padding:"5px 12px",borderRadius:7,fontSize:10,cursor:"pointer",
              fontFamily:"'Syne',sans-serif",fontWeight:600,
              background:"rgba(167,139,250,0.1)",border:"1px solid #A78BFA44",color:AI_COLOR,
            }}>🧠 AI Resume</button>
          </div>
          <div style={{
            display:"flex",alignItems:"center",gap:5,padding:"4px 10px",
            background:"rgba(255,255,255,0.03)",borderRadius:7,border:"1px solid rgba(255,255,255,0.05)",
          }}>
            <span style={{fontSize:11}}>{p?.plugged?"🔌":"🔋"}</span>
            <span style={{fontFamily:"'Space Mono',monospace",fontSize:11,color:p?.plugged?GPU_COLOR:NPU_COLOR}}>
              {fmt(p?.battery_pct,0)}%
            </span>
          </div>
        </div>

        {/* Main layout */}
        <div style={{display:"flex",gap:14,marginBottom:14}}>
          {/* Left column — video + stats */}
          <div style={{flex:"0 0 380px",display:"flex",flexDirection:"column",gap:10}}>
            {/* Video feed */}
            <div style={{position:"relative",borderRadius:10,overflow:"hidden",border:`1px solid ${activeColor}33`}}>
              {w?.frame_b64
                ? <img src={`data:image/jpeg;base64,${w.frame_b64}`} style={{width:"100%",display:"block"}} alt="AI"/>
                : <div style={{aspectRatio:"16/9",background:"#0a0a0a",display:"flex",alignItems:"center",justifyContent:"center",color:"#2a2a2a",fontSize:11}}>Waiting for camera…</div>
              }
              <div style={{
                position:"absolute",top:7,left:7,padding:"3px 9px",
                background:"rgba(0,0,0,0.75)",backdropFilter:"blur(6px)",
                border:`1px solid ${activeColor}55`,borderRadius:18,
                fontSize:9,color:activeColor,fontFamily:"'Space Mono',monospace",
              }}>
                {w?.processor} · LIVE AI
                {simulating&&<span style={{color:"#FF6B35",marginLeft:6}}>SIM</span>}
              </div>
            </div>

            {/* Stat cards */}
            <div style={{display:"flex",gap:8}}>
              {[
                ["FPS",fmt(w?.fps),"fps",activeColor],
                ["Latency",fmt(w?.inference_ms,0),"ms",activeColor],
                ["Power",fmt(w?.wattage),"W",activeColor],
                ["Battery",fmt(p?.battery_pct,0),"%",p?.plugged?GPU_COLOR:NPU_COLOR],
              ].map(([label,value,unit,color])=>(
                <div key={label} style={{
                  flex:1,background:"rgba(255,255,255,0.025)",
                  border:"1px solid rgba(255,255,255,0.05)",borderRadius:8,padding:"10px 12px",
                }}>
                  <div style={{fontSize:8,color:"#555",letterSpacing:"0.15em",textTransform:"uppercase",marginBottom:4}}>{label}</div>
                  <div style={{fontSize:22,fontFamily:"'Space Mono',monospace",color,lineHeight:1}}>
                    {value}<span style={{fontSize:10,color:"#444",marginLeft:2}}>{unit}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Spark charts */}
            <div style={{display:"flex",gap:8}}>
              <SparkChart history={history} dataKey="wattage" lo={0} hi={55} color={activeColor} label="Power" unit="W"/>
              <SparkChart history={history} dataKey="fps"     lo={0} hi={70} color={activeColor} label="FPS"   unit="fps"/>
            </div>
            <SparkChart history={history} dataKey="inf" lo={0} hi={200} color={AI_COLOR} label="Inference Latency" unit="ms"/>

            {/* Event log */}
            <div>
              <div style={{fontSize:8,color:"#3a3a3a",letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:4}}>
                System Event Log
              </div>
              <EventLog events={events}/>
            </div>
          </div>

          {/* Right column — AI + controls */}
          <div style={{flex:1,display:"flex",flexDirection:"column",gap:12}}>

            {/* Policy panel */}
            <PolicyPanel policy={pol} telemetry={t} policyStatus={policyStatus}/>

            {/* Live telemetry signals proof */}
            <TelemetryProof telemetry={t} simulating={simulating}/>

            {/* Simulate conditions */}
            <SimulatePanel simulating={simulating} onScenario={handleScenario} addEvent={addEvent}/>

            {/* Training panel */}
            <TrainingPanel
              policyStatus={policyStatus} telemetryRows={telemetryRows}
              onCollect={handleCollect} onTrain={handleTrain}
              collecting={collecting} training={training}
            />
          </div>
        </div>

        {/* Sustainability panel */}
        <SustainPanel sustain={s} benchmark={bench}/>

        {/* Footer */}
        <div style={{
          marginTop:14,paddingTop:12,borderTop:"1px solid rgba(255,255,255,0.04)",
          display:"flex",justifyContent:"space-between",fontSize:8,color:"#1a1a1a",
        }}>
          <span>EcoScale v3 · AMD Slingshot · Sustainable AI & Green Tech</span>
          <span>psutil · pynvml · WMI · ONNX Runtime · scikit-learn · React</span>
        </div>
      </div>
    </>
  );
}

