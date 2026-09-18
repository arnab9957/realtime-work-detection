import { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Grid, Text, Sphere, Box, Cylinder } from '@react-three/drei';
import type { OrbitControls as OrbitControlsType } from 'three/examples/jsm/controls/OrbitControls.js';
import * as THREE from 'three';
import { Package, LayoutGrid, Eye, RefreshCw } from 'lucide-react';
import type { DigitalTwinData } from '../../types/api';
import { useDigitalTwin } from '../../hooks/useDigitalTwin';


// ── Lerped mesh wrapper ───────────────────────────────────────────────────────
function LerpedMesh({ targetPos, children, alpha = 0.08 }: {
  targetPos: [number, number, number];
  children: React.ReactNode;
  alpha?: number;
}) {
  const ref = useRef<THREE.Group>(null);
  const target = useRef(new THREE.Vector3(...targetPos));
  useEffect(() => { target.current.set(...targetPos); }, [targetPos]);
  useFrame(() => { if (ref.current) ref.current.position.lerp(target.current, alpha); });
  return <group ref={ref}>{children}</group>;
}

// ── Container box + animated lid ─────────────────────────────────────────────
function Container({ data }: { data: DigitalTwinData }) {
  const lidRef = useRef<THREE.Group>(null);
  const targetLid = useRef(0);
  const { container } = data;

  useEffect(() => {
    targetLid.current = -THREE.MathUtils.degToRad(container.lid_angle_deg);
  }, [container.lid_angle_deg]);

  useFrame(() => {
    if (lidRef.current) {
      lidRef.current.rotation.x = THREE.MathUtils.lerp(
        lidRef.current.rotation.x, targetLid.current, 0.07
      );
    }
  });

  const [ox, oy, oz] = container.origin;
  return (
    <group position={[ox, oy, oz]}>
      {/* Body — neutral warm gray, technical material */}
      <Box args={[0.4, 0.3, 0.3]} position={[0, 0.15, 0]} castShadow receiveShadow>
        <meshStandardMaterial color="#C4C8CC" metalness={0.2} roughness={0.75} />
      </Box>
      {/* Interior base hint */}
      <Box args={[0.38, 0.01, 0.28]} position={[0, 0.005, 0]}>
        <meshStandardMaterial color="#B0B4B8" roughness={0.9} />
      </Box>
      {/* Lid */}
      <group ref={lidRef} position={[0, 0.3, -0.15]}>
        <Box args={[0.4, 0.018, 0.3]} position={[0, 0, 0.15]} castShadow>
          <meshStandardMaterial color="#CDD1D5" metalness={0.15} roughness={0.7} />
        </Box>
        {/* Lid edge highlight strip */}
        <Box args={[0.4, 0.005, 0.005]} position={[0, -0.009, 0.3]}>
          <meshStandardMaterial color="#A8AEBA" roughness={0.8} />
        </Box>
      </group>
    </group>
  );
}

// ── Experiment boxes (red / yellow) ──────────────────────────────────────────
function ObjectBox({ pos, color, edgeColor, label, showLabel }: {
  pos: [number, number, number];
  color: string;
  edgeColor: string;
  label: string;
  showLabel: boolean;
}) {
  return (
    <LerpedMesh targetPos={pos}>
      <Box args={[0.075, 0.075, 0.075]} castShadow>
        <meshStandardMaterial color={color} roughness={0.65} metalness={0.05} />
      </Box>
      {/* Edge outline effect using a slightly larger wireframe */}
      <Box args={[0.078, 0.078, 0.078]}>
        <meshBasicMaterial color={edgeColor} wireframe transparent opacity={0.25} />
      </Box>
      {showLabel && (
        <Text position={[0, 0.09, 0]} fontSize={0.038} color="#6B7280" anchorX="center" anchorY="bottom">
          {label}
        </Text>
      )}
    </LerpedMesh>
  );
}

// ── Human skeleton (joints + bones) ──────────────────────────────────────────
function HumanFigure({ joints, showJoints }: {
  joints: DigitalTwinData['astronaut']['joints'];
  showJoints: boolean;
}) {
  const scale = 2.5;

  function toScene(pos: [number, number, number]): [number, number, number] {
    return [pos[0] * scale, pos[1] * scale, pos[2] * scale];
  }

  const pairs: [string, string][] = [['shoulder', 'elbow'], ['elbow', 'wrist']];

  return (
    <group>
      {showJoints && Object.keys(joints).map(name => {
        const j = joints[name];
        const [sx, sy, sz] = toScene(j.pos_rack);
        return (
          <Sphere key={name} args={[0.022]} position={[sx, sy, sz]}>
            <meshStandardMaterial
              color="#2563EB"
              transparent opacity={Math.max(0.4, j.confidence)}
              roughness={0.4}
            />
          </Sphere>
        );
      })}

      {pairs.map(([a, b]) => {
        if (!joints[a] || !joints[b]) return null;
        const pa = new THREE.Vector3(...toScene(joints[a].pos_rack));
        const pb = new THREE.Vector3(...toScene(joints[b].pos_rack));
        return <BoneLine key={`${a}-${b}`} from={pa} to={pb} />;
      })}

      {/* Torso stub */}
      {joints.shoulder && (
        <LerpedMesh targetPos={toScene(joints.shoulder.pos_rack)}>
          <Cylinder args={[0.022, 0.022, 0.32]} rotation={[0, 0, 0]}>
            <meshStandardMaterial color="#94A3B8" roughness={0.7} />
          </Cylinder>
        </LerpedMesh>
      )}
    </group>
  );
}

// ── Bone line between two joints ──────────────────────────────────────────────
function BoneLine({ from, to }: { from: THREE.Vector3; to: THREE.Vector3 }) {
  const mid = from.clone().add(to).multiplyScalar(0.5);
  const dir = to.clone().sub(from);
  const len = dir.length();
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.normalize()
  );
  return (
    <Cylinder args={[0.007, 0.007, len]} position={[mid.x, mid.y, mid.z]} quaternion={quat}>
      <meshStandardMaterial color="#64748B" transparent opacity={0.55} roughness={0.6} />
    </Cylinder>
  );
}

// ── Main scene ────────────────────────────────────────────────────────────────
function Scene({ data, showGrid, showLabels, showJoints }: {
  data: DigitalTwinData;
  showGrid: boolean;
  showLabels: boolean;
  showJoints: boolean;
}) {
  const compBox = data.entities?.component_box;
  const contBox = data.entities?.container_box;

  return (
    <>
      {/* Clean engineering lighting */}
      <ambientLight intensity={1.6} color="#F8FAFC" />
      <directionalLight
        position={[3, 5, 4]}
        intensity={0.9}
        color="#FFFFFF"
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-near={0.5}
        shadow-camera-far={12}
      />
      <directionalLight position={[-2, 3, -2]} intensity={0.3} color="#E0E8F0" />
      <pointLight position={[0, 2, 2]} intensity={0.15} color="#CBD5E1" />

      {/* Floor — neutral cool gray */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[5, 5]} />
        <meshStandardMaterial color="#E8ECF0" roughness={0.9} metalness={0} />
      </mesh>

      {/* Grid — very subtle */}
      {showGrid && (
        <Grid
          args={[5, 5]}
          position={[0, 0.001, 0]}
          cellColor="#D1D5DB"
          sectionColor="#C4C8CC"
          cellSize={0.2}
          sectionSize={1}
          fadeDistance={5}
          fadeStrength={1.5}
        />
      )}

      {/* Rack bounding box — thin wireframe reference */}
      <mesh position={[0, 0.5, 0]}>
        <boxGeometry args={[1.0, 1.0, 1.2]} />
        <meshBasicMaterial color="#CBD5E1" wireframe transparent opacity={0.35} />
      </mesh>

      {/* Container */}
      <Container data={data} />

      {/* Red experiment box */}
      {compBox && (
        <ObjectBox
          pos={compBox.pos_rack}
          color="#EF4444"
          edgeColor="#B91C1C"
          label="RED"
          showLabel={showLabels}
        />
      )}

      {/* Yellow experiment box */}
      {contBox && (
        <ObjectBox
          pos={contBox.pos_rack}
          color="#EAB308"
          edgeColor="#A16207"
          label="CONTAINER"
          showLabel={showLabels}
        />
      )}

      {/* Human skeleton */}
      {data.astronaut?.joints && (
        <HumanFigure joints={data.astronaut.joints} showJoints={showJoints} />
      )}

      {/* Rack label */}
      {showLabels && (
        <Text position={[0, -0.06, 0.68]} fontSize={0.042} color="#9CA3AF" anchorX="center">
          {data.rack?.label || 'BAS-03 Payload Rack'}
        </Text>
      )}
    </>
  );
}

// ── Camera reset ──────────────────────────────────────────────────────────────
function CameraResetter({ trigger }: { trigger: number }) {
  const { camera } = useThree();
  const controls = useThree(state => state.controls) as OrbitControlsType | null;
  useEffect(() => {
    if (trigger === 0) return;
    camera.position.set(1.4, 1.1, 1.7);
    camera.lookAt(0, 0.5, 0);
    if (controls) controls.target.set(0, 0.4, 0);
  }, [trigger, camera, controls]);
  return null;
}

// ── Camera2 panel ─────────────────────────────────────────────────────────────
export function Camera2() {
  const { data } = useDigitalTwin();
  const [showGrid,   setShowGrid]   = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showJoints, setShowJoints] = useState(true);
  const [resetTrigger, setResetTrigger] = useState(0);

  return (
    <div className="panel camera2-panel">
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Package size={13} strokeWidth={1.8} style={{ color: 'var(--text-muted)' }} />
          <span className="panel-title">Camera 02 — HMR 3D Visualization</span>
        </div>
        <div className="viewer-controls">
          <Tog active={showGrid}   onClick={() => setShowGrid(g => !g)}   icon={<LayoutGrid size={11} strokeWidth={1.8} />} label="Grid" />
          <Tog active={showLabels} onClick={() => setShowLabels(l => !l)} icon={<Eye size={11} strokeWidth={1.8} />} label="Labels" />
          <Tog active={showJoints} onClick={() => setShowJoints(j => !j)} icon={<Eye size={11} strokeWidth={1.8} />} label="Joints" />
          <button className="ctrl-btn ctrl-btn-sm" onClick={() => setResetTrigger(t => t + 1)}
            style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <RefreshCw size={11} strokeWidth={2} /> Reset View
          </button>
        </div>
      </div>

      <div className="canvas-wrapper">
        <Canvas
          camera={{ position: [1.4, 1.1, 1.7], fov: 48 }}
          shadows={{ type: THREE.PCFShadowMap }}
          gl={{ antialias: true, alpha: false }}
          onCreated={({ gl }) => { gl.setClearColor('#EDF0F4', 1); }}
        >
          <CameraResetter trigger={resetTrigger} />
          <OrbitControls
            makeDefault
            minDistance={0.5} maxDistance={6}
            target={[0, 0.4, 0]}
            enableDamping dampingFactor={0.07}
          />
          <Scene
            data={data}
            showGrid={showGrid}
            showLabels={showLabels}
            showJoints={showJoints}
          />
        </Canvas>

        {/* HUD */}
        <div className="viewer-hud">
          <span className="hud-chip">ELBOW {data.astronaut?.elbow_angle_deg?.toFixed(1) ?? '—'}°</span>
          <span className={`hud-chip ${data.astronaut?.rom_violated ? 'hud-warn' : ''}`}>
            ROM {data.astronaut?.rom_violated ? '⚠ VIOLATED' : '✓ OK'}
          </span>
          <span className="hud-chip">LID {data.container?.lid_angle_deg?.toFixed(1) ?? '—'}°</span>
          <span className="hud-chip">{data.step || 'IDLE'}</span>
        </div>
      </div>
    </div>
  );
}

function Tog({ active, onClick, icon, label }: {
  active: boolean; onClick: () => void;
  icon: React.ReactNode; label: string;
}) {
  return (
    <button
      className={`toggle-btn ${active ? 'toggle-active' : ''}`}
      onClick={onClick}
      style={{ display: 'flex', alignItems: 'center', gap: 4 }}
    >
      {icon} {label}
    </button>
  );
}
