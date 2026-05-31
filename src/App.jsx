import Activity from "lucide-react/dist/esm/icons/activity.mjs";
import ExternalLink from "lucide-react/dist/esm/icons/external-link.mjs";
import Layers3 from "lucide-react/dist/esm/icons/layers-3.mjs";
import Loader2 from "lucide-react/dist/esm/icons/loader-2.mjs";
import LocateFixed from "lucide-react/dist/esm/icons/locate-fixed.mjs";
import Radio from "lucide-react/dist/esm/icons/radio.mjs";
import Send from "lucide-react/dist/esm/icons/send.mjs";
import Sparkles from "lucide-react/dist/esm/icons/sparkles.mjs";
import Tags from "lucide-react/dist/esm/icons/tags.mjs";
import Target from "lucide-react/dist/esm/icons/target.mjs";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const OUTLET_STYLES = {
  "Fox News": { color: "#ff4f64", short: "FOX" },
  Breitbart: { color: "#ff9a3d", short: "BR" },
  NYT: { color: "#51a7ff", short: "NYT" },
  "The Guardian": { color: "#2dd4bf", short: "GDN" },
  "NBC News": { color: "#ffe25e", short: "NBC" },
  "Washington Post": { color: "#d5dbe8", short: "WP" },
  NPR: { color: "#b6f35e", short: "NPR" },
};

const DEFAULT_COLOR = "#9aa8bc";

const STOP_WORDS = new Set([
  "about",
  "after",
  "again",
  "against",
  "also",
  "because",
  "before",
  "being",
  "between",
  "could",
  "during",
  "every",
  "from",
  "have",
  "into",
  "more",
  "most",
  "over",
  "said",
  "should",
  "than",
  "that",
  "their",
  "there",
  "these",
  "they",
  "this",
  "through",
  "under",
  "when",
  "where",
  "which",
  "while",
  "with",
  "would",
]);

const PROFILE_TERMS = {
  "Fox News": [
    "border",
    "crime",
    "freedom",
    "parents",
    "taxpayer",
    "biden",
    "democrat",
    "liberal",
    "woke",
    "illegal",
  ],
  Breitbart: [
    "establishment",
    "globalist",
    "migration",
    "populist",
    "elite",
    "media",
    "censorship",
    "sovereignty",
    "leftist",
  ],
  NYT: [
    "institutions",
    "democracy",
    "officials",
    "policy",
    "administration",
    "court",
    "climate",
    "rights",
  ],
  "The Guardian": [
    "inequality",
    "climate",
    "justice",
    "workers",
    "far-right",
    "progressive",
    "crisis",
    "activists",
  ],
  "NBC News": [
    "breaking",
    "officials",
    "campaign",
    "voters",
    "senate",
    "white house",
    "police",
    "investigation",
  ],
  "Washington Post": [
    "accountability",
    "power",
    "sources",
    "analysis",
    "federal",
    "washington",
    "oversight",
    "president",
  ],
  NPR: [
    "community",
    "heard",
    "interview",
    "local",
    "conversation",
    "program",
    "public",
    "culture",
  ],
};

function outletColor(outlet) {
  return OUTLET_STYLES[outlet]?.color || DEFAULT_COLOR;
}

function computeBounds(points) {
  const axes = ["x", "y", "z"];
  const bounds = {};
  for (const axis of axes) {
    const values = points.map((point) => Number(point[axis] || 0));
    const min = Math.min(...values);
    const max = Math.max(...values);
    bounds[axis] = { min, max, center: (min + max) / 2 };
  }
  return bounds;
}

function createProjector(points) {
  if (!points.length) {
    return () => new THREE.Vector3(0, 0, 0);
  }

  const bounds = computeBounds(points);
  return (point) =>
    new THREE.Vector3(
      (Number(point.x) - bounds.x.center) * 1.55,
      (Number(point.z) - bounds.z.center) * 0.9,
      (Number(point.y) - bounds.y.center) * 1.05,
    );
}

function hashText(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function extractTerms(text, limit = 8) {
  const counts = new Map();
  const words = text.toLowerCase().match(/[a-z][a-z-]{3,}/g) || [];
  for (const word of words) {
    if (STOP_WORDS.has(word)) continue;
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([word]) => word);
}

function normalizedSimilarities(coords, centroids) {
  const entries = Object.entries(centroids).map(([outlet, centroid]) => {
    const distance = Math.hypot(
      coords.x - centroid.x,
      coords.y - centroid.y,
      coords.z - centroid.z,
    );
    return [outlet, 1 / (0.25 + distance)];
  });
  const total = entries.reduce((sum, [, score]) => sum + score, 0) || 1;
  return Object.fromEntries(
    entries.map(([outlet, score]) => [outlet, Number(((score / total) * 100).toFixed(1))]),
  );
}

function nearestByCoordinate(coords, points, limit = 5) {
  return [...points]
    .map((point) => ({
      headline: point.headline,
      outlet: point.outlet,
      url: point.url,
      text_preview: point.text_preview,
      distance: Math.hypot(coords.x - point.x, coords.y - point.y, coords.z - point.z),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, limit);
}

function buildFallbackResult(text, points, centroids) {
  const lower = text.toLowerCase();
  const outletNames = Object.keys(centroids);
  const rawScores = Object.fromEntries(outletNames.map((outlet) => [outlet, 1]));

  for (const [outlet, terms] of Object.entries(PROFILE_TERMS)) {
    for (const term of terms) {
      const needle = term.toLowerCase();
      const count = lower.split(needle).length - 1;
      rawScores[outlet] = (rawScores[outlet] || 1) + count * 1.9;
    }
  }

  const maxScore = Math.max(...Object.values(rawScores));
  const weights = Object.fromEntries(
    Object.entries(rawScores).map(([outlet, score]) => [
      outlet,
      Math.exp((score - maxScore) / 2.2),
    ]),
  );
  const totalWeight = Object.values(weights).reduce((sum, value) => sum + value, 0) || 1;

  const coords = { x: 0, y: 0, z: 0 };
  for (const [outlet, weight] of Object.entries(weights)) {
    const centroid = centroids[outlet];
    if (!centroid) continue;
    const normalized = weight / totalWeight;
    coords.x += centroid.x * normalized;
    coords.y += centroid.y * normalized;
    coords.z += centroid.z * normalized;
  }

  const hash = hashText(text);
  coords.x += ((hash % 997) / 997 - 0.5) * 0.7;
  coords.y += (((hash >> 8) % 991) / 991 - 0.5) * 0.55;
  coords.z += (((hash >> 16) % 983) / 983 - 0.5) * 0.7;

  const similarities = normalizedSimilarities(coords, centroids);
  const sorted = Object.entries(similarities).sort((a, b) => b[1] - a[1]);
  const terms = extractTerms(text);

  return {
    ...coords,
    similarities,
    nearest: nearestByCoordinate(coords, points),
    distinctive_terms: terms,
    top_outlet: sorted[0]?.[0] || "Unknown",
    top_score: sorted[0]?.[1] || 0,
    interpretation:
      "Demo placement used the bundled corpus and a lexical fallback. The Modal backend replaces this with sentence-transformer projection and an OpenRouter read.",
    mode: "frontend-demo",
  };
}

function normalizeEmbedResult(data, text, points, centroids) {
  function normalizeView(view = data) {
    const coords = {
      x: Number(view.x ?? view.coords?.x ?? data.x ?? 0),
      y: Number(view.y ?? view.coords?.y ?? data.y ?? 0),
      z: Number(view.z ?? view.coords?.z ?? data.z ?? 0),
    };
    return {
      ...coords,
      similarities: view.similarities || data.similarities || normalizedSimilarities(coords, centroids),
      nearest: view.nearest?.length ? view.nearest : data.nearest || nearestByCoordinate(coords, points),
      projection: view.projection || data.projection || "",
    };
  }

  const worldview = normalizeView(data.views?.worldview || data);
  const editorial = normalizeView(data.views?.editorial || data);

  return {
    ...worldview,
    views: {
      worldview,
      editorial,
    },
    nearest: data.nearest?.length ? data.nearest : editorial.nearest,
    framing_nearest: data.framing_nearest?.length ? data.framing_nearest : data.nearest || [],
    distinctive_terms: data.distinctive_terms?.length
      ? data.distinctive_terms
      : extractTerms(text),
    topic: data.topic || null,
    top_outlet: data.top_outlet || "Unknown",
    top_score: Number(data.top_score ?? 0),
    interpretation: data.interpretation || "",
    mode: data.mode || "modal-backend",
  };
}

function createTopicLabelSprite(label, count) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const title = label.toUpperCase();
  const subtitle = `${count} ARTICLES`;
  context.font = `700 ${15 * pixelRatio}px Inter, sans-serif`;
  const textWidth = Math.max(
    context.measureText(title).width,
    context.measureText(subtitle).width,
  );
  canvas.width = Math.ceil(textWidth + 30 * pixelRatio);
  canvas.height = Math.ceil(52 * pixelRatio);
  context.scale(pixelRatio, pixelRatio);
  const width = canvas.width / pixelRatio;
  const height = canvas.height / pixelRatio;

  context.fillStyle = "rgba(5, 11, 17, 0.74)";
  context.strokeStyle = "rgba(114, 244, 255, 0.42)";
  context.lineWidth = 1;
  context.beginPath();
  context.roundRect(0.5, 0.5, width - 1, height - 1, 6);
  context.fill();
  context.stroke();

  context.fillStyle = "#dffbff";
  context.font = "700 15px Inter, sans-serif";
  context.fillText(title, 14, 22);
  context.fillStyle = "rgba(168, 190, 211, 0.9)";
  context.font = "600 10px Inter, sans-serif";
  context.fillText(subtitle, 14, 39);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: 0.48,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(Math.max(2.8, width / 52), height / 52, 1);
  return sprite;
}

function WorldviewScene({
  points,
  centroids,
  visibleOutlets,
  topicRegions,
  showTopicLabels,
  userResult,
  selectedPoint,
  focusResetVersion,
  onHover,
  onSelect,
}) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const pointsGroupRef = useRef(null);
  const centroidGroupRef = useRef(null);
  const topicGroupRef = useRef(null);
  const userGroupRef = useRef(null);
  const pointObjectsRef = useRef([]);
  const focusAnimationRef = useRef(null);
  const projectorRef = useRef(createProjector([]));
  const onHoverRef = useRef(onHover);
  const onSelectRef = useRef(onSelect);
  const visibleRef = useRef(visibleOutlets);

  useEffect(() => {
    onHoverRef.current = onHover;
    onSelectRef.current = onSelect;
    visibleRef.current = visibleOutlets;
  }, [onHover, onSelect, visibleOutlets]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#05070b");
    scene.fog = new THREE.FogExp2("#05070b", 0.035);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(48, mount.clientWidth / mount.clientHeight, 0.1, 120);
    camera.position.set(7.8, 5.3, 12.2);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 4;
    controls.maxDistance = 28;
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;

    function animateFocus(target, cameraOffset = new THREE.Vector3(5.3, 3.1, 6.7)) {
      focusAnimationRef.current = {
        target: target.clone(),
        cameraPosition: target.clone().add(cameraOffset),
      };
    }

    function cancelFocus() {
      focusAnimationRef.current = null;
    }

    const ambient = new THREE.AmbientLight("#8aa4c8", 0.85);
    scene.add(ambient);
    const key = new THREE.DirectionalLight("#ffffff", 2.2);
    key.position.set(3, 8, 5);
    scene.add(key);
    const rim = new THREE.PointLight("#34d6ff", 4.8, 28);
    rim.position.set(-7, 3, -5);
    scene.add(rim);

    const grid = new THREE.GridHelper(22, 22, "#273142", "#151c29");
    grid.position.y = -5.25;
    grid.material.transparent = true;
    grid.material.opacity = 0.22;
    scene.add(grid);

    const starGeometry = new THREE.BufferGeometry();
    const starPositions = [];
    for (let index = 0; index < 760; index += 1) {
      starPositions.push(
        (Math.random() - 0.5) * 54,
        (Math.random() - 0.5) * 32,
        (Math.random() - 0.5) * 54,
      );
    }
    starGeometry.setAttribute("position", new THREE.Float32BufferAttribute(starPositions, 3));
    const stars = new THREE.Points(
      starGeometry,
      new THREE.PointsMaterial({
        color: "#d9e8ff",
        size: 0.022,
        transparent: true,
        opacity: 0.48,
        depthWrite: false,
      }),
    );
    scene.add(stars);

    const pointsGroup = new THREE.Group();
    pointsGroupRef.current = pointsGroup;
    scene.add(pointsGroup);

    const centroidGroup = new THREE.Group();
    centroidGroupRef.current = centroidGroup;
    scene.add(centroidGroup);

    const topicGroup = new THREE.Group();
    topicGroupRef.current = topicGroup;
    scene.add(topicGroup);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    function handlePointerMove(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const visibleObjects = pointObjectsRef.current.filter((object) => object.visible);
      const [hit] = raycaster.intersectObjects(visibleObjects, false);
      renderer.domElement.style.cursor = hit ? "pointer" : "grab";
      onHoverRef.current(
        hit
          ? {
              point: hit.object.userData.point,
              x: event.clientX,
              y: event.clientY,
            }
          : null,
      );
    }

    function handleClick() {
      const visibleObjects = pointObjectsRef.current.filter((object) => object.visible);
      const [hit] = raycaster.intersectObjects(visibleObjects, false);
      if (hit) {
        onSelectRef.current(hit.object.userData.point);
        animateFocus(hit.object.position);
      }
    }

    function handleResize() {
      if (!mount.clientWidth || !mount.clientHeight) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    }

    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("click", handleClick);
    controls.addEventListener("start", cancelFocus);
    window.addEventListener("resize", handleResize);

    let animationId = 0;
    function animate() {
      const elapsed = performance.now() * 0.001;
      stars.rotation.y = elapsed * 0.012;

      for (let index = 0; index < pointObjectsRef.current.length; index += 1) {
        const point = pointObjectsRef.current[index];
        const pulse = 1 + Math.sin(elapsed * 1.35 + index * 0.31) * 0.035;
        point.scale.setScalar(pulse);
      }

      if (userGroupRef.current) {
        const target = userGroupRef.current.userData.target;
        if (target) {
          userGroupRef.current.position.lerp(target, 0.055);
        }
        userGroupRef.current.rotation.y += 0.018;
        userGroupRef.current.children.forEach((child, index) => {
          if (child.material?.opacity && index > 0) {
            child.material.opacity = 0.18 + Math.sin(elapsed * 2.4 + index) * 0.045;
          }
        });
      }

      for (const label of topicGroupRef.current?.children || []) {
        const distance = camera.position.distanceTo(label.position);
        label.material.opacity = THREE.MathUtils.clamp(0.74 - distance / 40, 0.22, 0.62);
      }

      if (focusAnimationRef.current) {
        const { target, cameraPosition } = focusAnimationRef.current;
        controls.target.lerp(target, 0.075);
        camera.position.lerp(cameraPosition, 0.055);
        if (
          controls.target.distanceTo(target) < 0.025 &&
          camera.position.distanceTo(cameraPosition) < 0.04
        ) {
          controls.target.copy(target);
          camera.position.copy(cameraPosition);
          focusAnimationRef.current = null;
        }
      }

      controls.update();
      renderer.render(scene, camera);
      animationId = requestAnimationFrame(animate);
    }
    animate();

    return () => {
      cancelAnimationFrame(animationId);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("click", handleClick);
      controls.removeEventListener("start", cancelFocus);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    const pointsGroup = pointsGroupRef.current;
    const centroidGroup = centroidGroupRef.current;
    const topicGroup = topicGroupRef.current;
    if (!scene || !pointsGroup || !centroidGroup || !topicGroup || !points.length) return;

    projectorRef.current = createProjector(points);

    pointsGroup.clear();
    centroidGroup.clear();
    topicGroup.clear();
    pointObjectsRef.current = [];

    const sphere = new THREE.SphereGeometry(0.075, 16, 16);
    for (const point of points) {
      const color = outletColor(point.outlet);
      const material = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.52,
        roughness: 0.42,
        metalness: 0.18,
        transparent: true,
        opacity: 0.76,
      });
      const mesh = new THREE.Mesh(sphere, material);
      mesh.position.copy(projectorRef.current(point));
      mesh.userData.point = point;
      mesh.visible = visibleRef.current.has(point.outlet);
      pointsGroup.add(mesh);
      pointObjectsRef.current.push(mesh);
    }

    const centroidGeometry = new THREE.TorusGeometry(0.28, 0.016, 12, 44);
    for (const [outlet, centroid] of Object.entries(centroids)) {
      const color = outletColor(outlet);
      const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
      });
      const marker = new THREE.Mesh(centroidGeometry, material);
      marker.position.copy(projectorRef.current(centroid));
      marker.rotation.x = Math.PI / 2;
      marker.userData.outlet = outlet;
      centroidGroup.add(marker);
    }

    if (showTopicLabels) {
      for (const topic of topicRegions) {
        const label = createTopicLabelSprite(topic.label, topic.count);
        label.position.copy(projectorRef.current(topic.editorial));
        label.position.y += 0.44;
        topicGroup.add(label);
      }
    }
  }, [points, centroids, topicRegions, showTopicLabels]);

  useEffect(() => {
    for (const object of pointObjectsRef.current) {
      object.visible = visibleOutlets.has(object.userData.point.outlet);
    }
    for (const object of centroidGroupRef.current?.children || []) {
      object.visible = visibleOutlets.has(object.userData.outlet);
    }
  }, [visibleOutlets]);

  useEffect(() => {
    if (!selectedPoint) return;
    const object = pointObjectsRef.current.find(
      (mesh) => mesh.userData.point.url === selectedPoint.url,
    );
    if (object) {
      focusAnimationRef.current = {
        target: object.position.clone(),
        cameraPosition: object.position.clone().add(new THREE.Vector3(5.3, 3.1, 6.7)),
      };
    }
  }, [selectedPoint]);

  useEffect(() => {
    focusAnimationRef.current = {
      target: new THREE.Vector3(0, 0, 0),
      cameraPosition: new THREE.Vector3(7.8, 5.3, 12.2),
    };
  }, [focusResetVersion]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !userResult) return;

    if (userGroupRef.current) {
      scene.remove(userGroupRef.current);
    }

    const target = projectorRef.current(userResult);
    const group = new THREE.Group();
    group.position.copy(target.clone().add(new THREE.Vector3(0, 8, 0)));
    group.userData.target = target;

    const core = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.22, 1),
      new THREE.MeshStandardMaterial({
        color: "#ffffff",
        emissive: "#76f7ff",
        emissiveIntensity: 1.8,
        roughness: 0.18,
        metalness: 0.35,
      }),
    );
    group.add(core);

    const glow = new THREE.Mesh(
      new THREE.SphereGeometry(0.62, 32, 32),
      new THREE.MeshBasicMaterial({
        color: "#76f7ff",
        transparent: true,
        opacity: 0.2,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    group.add(glow);

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.5, 0.012, 10, 58),
      new THREE.MeshBasicMaterial({
        color: "#ffe25e",
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
      }),
    );
    ring.rotation.x = Math.PI / 2;
    group.add(ring);

    scene.add(group);
    userGroupRef.current = group;
    focusAnimationRef.current = {
      target: target.clone(),
      cameraPosition: target.clone().add(new THREE.Vector3(5.3, 3.1, 6.7)),
    };
  }, [userResult]);

  return <div className="scene-canvas" ref={mountRef} />;
}

export default function App() {
  const [points, setPoints] = useState([]);
  const [centroids, setCentroids] = useState({});
  const [worldviewPoints, setWorldviewPoints] = useState([]);
  const [worldviewCentroids, setWorldviewCentroids] = useState({});
  const [topics, setTopics] = useState([]);
  const [viewMode, setViewMode] = useState("worldview");
  const [showTopicLabels, setShowTopicLabels] = useState(true);
  const [visibleOutlets, setVisibleOutlets] = useState(new Set());
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [text, setText] = useState("");
  const [userResult, setUserResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [analysisError, setAnalysisError] = useState("");
  const [focusResetVersion, setFocusResetVersion] = useState(0);

  useEffect(() => {
    let cancelled = false;
    async function loadData() {
      try {
        const [
          pointsResponse,
          centroidsResponse,
          worldviewPointsResponse,
          worldviewCentroidsResponse,
          topicsResponse,
        ] = await Promise.all([
          fetch("/worldview/points.json"),
          fetch("/worldview/centroids.json"),
          fetch("/worldview/worldview_points.json"),
          fetch("/worldview/worldview_centroids.json"),
          fetch("/worldview/topics.json"),
        ]);
        if (
          !pointsResponse.ok ||
          !centroidsResponse.ok ||
          !worldviewPointsResponse.ok ||
          !worldviewCentroidsResponse.ok ||
          !topicsResponse.ok
        ) {
          throw new Error("Could not load worldview data.");
        }
        const [
          pointsData,
          centroidData,
          worldviewPointsData,
          worldviewCentroidData,
          topicsData,
        ] = await Promise.all([
          pointsResponse.json(),
          centroidsResponse.json(),
          worldviewPointsResponse.json(),
          worldviewCentroidsResponse.json(),
          topicsResponse.json(),
        ]);
        if (!cancelled) {
          setPoints(pointsData);
          setCentroids(centroidData);
          setWorldviewPoints(worldviewPointsData);
          setWorldviewCentroids(worldviewCentroidData);
          setTopics(topicsData);
          setVisibleOutlets(new Set([...new Set(pointsData.map((point) => point.outlet))]));
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(error.message || "Could not load worldview data.");
        }
      }
    }
    loadData();
    return () => {
      cancelled = true;
    };
  }, []);

  const outletStats = useMemo(() => {
    const counts = new Map();
    for (const point of points) {
      counts.set(point.outlet, (counts.get(point.outlet) || 0) + 1);
    }
    return [...counts.entries()]
      .map(([outlet, count]) => ({ outlet, count }))
      .sort((a, b) => b.count - a.count || a.outlet.localeCompare(b.outlet));
  }, [points]);

  const activePoints = viewMode === "worldview" && worldviewPoints.length ? worldviewPoints : points;
  const activeCentroids =
    viewMode === "worldview" && Object.keys(worldviewCentroids).length
      ? worldviewCentroids
      : centroids;
  const activeUserView = userResult?.views?.[viewMode] || userResult;

  const topSimilarity = useMemo(() => {
    if (!activeUserView?.similarities) return null;
    return Object.entries(activeUserView.similarities).sort((a, b) => b[1] - a[1])[0] || null;
  }, [activeUserView]);

  function toggleOutlet(outlet) {
    setVisibleOutlets((current) => {
      const next = new Set(current);
      if (next.has(outlet)) next.delete(outlet);
      else next.add(outlet);
      return next;
    });
  }

  function resetCamera() {
    setSelectedPoint(null);
    setFocusResetVersion((version) => version + 1);
  }

  async function handleAnalyze(event) {
    event.preventDefault();
    const trimmed = text.trim();
    if (trimmed.length < 40) {
      setAnalysisError("Add a fuller paragraph so the placement has enough signal.");
      return;
    }

    setAnalysisError("");
    setIsAnalyzing(true);
    try {
      let result;
      if (API_BASE) {
        const response = await fetch(`${API_BASE}/embed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: trimmed, interpret: true }),
        });
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }
        result = normalizeEmbedResult(await response.json(), trimmed, points, centroids);
      } else {
        result = buildFallbackResult(trimmed, points, centroids);
      }
      setUserResult(result);
      setSelectedPoint(null);
    } catch (error) {
      const fallback = buildFallbackResult(trimmed, points, centroids);
      setUserResult({
        ...fallback,
        mode: "frontend-demo",
        interpretation:
          "The deployed ML endpoint was unavailable, so this public demo used the bundled fallback placement.",
      });
      setAnalysisError(error.message || "Backend unavailable; showing demo placement.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  const visibleCount = activePoints.filter((point) => visibleOutlets.has(point.outlet)).length;

  return (
    <main className="app-shell">
      <WorldviewScene
        points={activePoints}
        centroids={activeCentroids}
        visibleOutlets={visibleOutlets}
        topicRegions={topics}
        showTopicLabels={viewMode === "editorial" && showTopicLabels}
        userResult={activeUserView}
        selectedPoint={selectedPoint}
        focusResetVersion={focusResetVersion}
        onHover={setHovered}
        onSelect={setSelectedPoint}
      />

      <section className="left-dock panel">
        <div className="brand-row">
          <span className="brand-mark">
            <Activity size={18} />
          </span>
          <div>
            <h1>Worldview Space</h1>
            <p>{points.length || "..."} articles projected into UMAP-3D</p>
          </div>
        </div>

        <div className="metric-grid">
          <div>
            <span>{outletStats.length || "-"}</span>
            <label>Outlets</label>
          </div>
          <div>
            <span>{visibleCount || "-"}</span>
            <label>Visible</label>
          </div>
        </div>

        <div className="dock-section">
          <div className="section-title">
            <Layers3 size={14} />
            Lens
          </div>
          <div className="segment-control">
            <button
              className={viewMode === "worldview" ? "is-active" : ""}
              onClick={() => setViewMode("worldview")}
              type="button"
            >
              Worldview
            </button>
            <button
              className={viewMode === "editorial" ? "is-active" : ""}
              onClick={() => setViewMode("editorial")}
              type="button"
            >
              Landscape
            </button>
          </div>
          {viewMode === "editorial" ? (
            <label className="toggle-row">
              <span>
                <Tags size={14} />
                Topic labels
              </span>
              <input
                checked={showTopicLabels}
                onChange={(event) => setShowTopicLabels(event.target.checked)}
                type="checkbox"
              />
              <i />
            </label>
          ) : null}
        </div>

        <div className="dock-section">
          <div className="section-title">
            <Radio size={14} />
            Corpus
          </div>
          <div className="legend-list">
            {outletStats.map(({ outlet, count }) => (
              <button
                className={`legend-row ${visibleOutlets.has(outlet) ? "is-visible" : ""}`}
                key={outlet}
                onClick={() => toggleOutlet(outlet)}
                type="button"
              >
                <span
                  className="swatch"
                  style={{ "--outlet-color": outletColor(outlet) }}
                  aria-hidden="true"
                />
                <span className="legend-name">{outlet}</span>
                <span className="legend-count">{count}</span>
              </button>
            ))}
          </div>
        </div>

        <button className="icon-action" onClick={resetCamera} type="button">
          <LocateFixed size={16} />
          Recenter
        </button>
      </section>

      <section className="right-dock panel">
        <div className="dock-header">
          <div>
            <h2>Submit Writing</h2>
            <p>{API_BASE ? "Modal backend connected" : "Public demo mode"}</p>
          </div>
          <span className={`status-dot ${API_BASE ? "live" : ""}`} />
        </div>

        <form className="analysis-form" onSubmit={handleAnalyze}>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Paste a paragraph of political writing..."
            spellCheck="true"
          />
          <div className="form-actions">
            <span>{text.trim().split(/\s+/).filter(Boolean).length} words</span>
            <button disabled={isAnalyzing || !points.length} type="submit">
              {isAnalyzing ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
              Analyze
            </button>
          </div>
          {analysisError ? <p className="form-error">{analysisError}</p> : null}
        </form>

        {userResult ? (
          <div className="result-panel">
            <div className="result-kicker">
              <Sparkles size={15} />
              Your point
            </div>
            <div className="top-read">
              <span>
                {viewMode === "worldview" && userResult.topic
                  ? `Among writing about ${userResult.topic.label}, your framing is closest to`
                  : "Closest outlet cluster"}
              </span>
              <strong style={{ color: outletColor(topSimilarity?.[0]) }}>
                {topSimilarity?.[0] || userResult.top_outlet}
              </strong>
              <b>
                {Number(topSimilarity?.[1] || userResult.top_score).toFixed(1)}%
                {viewMode === "worldview" ? " topic-conditioned match" : " cluster match"}
              </b>
            </div>

            <div className="similarity-bars">
              {Object.entries(activeUserView.similarities)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([outlet, score]) => (
                  <div className="bar-row" key={outlet}>
                    <span>{OUTLET_STYLES[outlet]?.short || outlet}</span>
                    <div className="bar-track">
                      <i
                        style={{
                          width: `${Math.max(4, Number(score))}%`,
                          background: outletColor(outlet),
                        }}
                      />
                    </div>
                    <em>{Number(score).toFixed(1)}</em>
                  </div>
                ))}
            </div>

            {userResult.distinctive_terms?.length ? (
              <div className="term-row">
                {userResult.distinctive_terms.slice(0, 6).map((term) => (
                  <span key={term}>{term}</span>
                ))}
              </div>
            ) : null}

            {userResult.nearest?.[0] ? (
              <a
                className="closest-article"
                href={userResult.nearest[0].url}
                target="_blank"
                rel="noreferrer"
              >
                <span>Closest individual article</span>
                <strong>{userResult.nearest[0].outlet}</strong>
                <em>{userResult.nearest[0].headline}</em>
              </a>
            ) : null}

            {userResult.interpretation ? (
              <p className="interpretation">{userResult.interpretation}</p>
            ) : null}
          </div>
        ) : null}

        <div className="article-panel">
          <div className="section-title">
            <Target size={14} />
            {selectedPoint
              ? "Selected Article"
              : viewMode === "worldview"
                ? "Similar Framing"
                : "Nearest Articles"}
          </div>
          {selectedPoint ? (
            <ArticleDetail point={selectedPoint} />
          ) : activeUserView?.nearest?.length ? (
            <div className="nearest-list">
              {activeUserView.nearest.slice(0, 4).map((point) => (
                <ArticleRow key={`${point.outlet}-${point.url}`} point={point} />
              ))}
            </div>
          ) : (
            <p className="empty-state">Click a point or submit writing to inspect the space.</p>
          )}
        </div>
      </section>

      <div className="bottom-strip">
        {viewMode === "worldview" ? (
          <>
            <span>Topic effects reduced</span>
            <span>Compare framing within subject matter</span>
          </>
        ) : (
          <>
            <span>Semantic proximity</span>
            <span>Topic and editorial framing coexist</span>
          </>
        )}
      </div>

      {hovered ? (
        <div className="tooltip" style={{ left: hovered.x + 14, top: hovered.y + 14 }}>
          <strong style={{ color: outletColor(hovered.point.outlet) }}>{hovered.point.outlet}</strong>
          <span>{hovered.point.headline}</span>
          {hovered.point.topic_label ? <em>{hovered.point.topic_label}</em> : null}
        </div>
      ) : null}

      {loadError ? <div className="load-error">{loadError}</div> : null}
    </main>
  );
}

function ArticleDetail({ point }) {
  return (
    <article className="article-detail">
      <span style={{ color: outletColor(point.outlet) }}>{point.outlet}</span>
      <h3>{point.headline}</h3>
      {point.text_preview ? <p>{point.text_preview}</p> : null}
      {point.url ? (
        <a href={point.url} target="_blank" rel="noreferrer">
          Open source
          <ExternalLink size={13} />
        </a>
      ) : null}
    </article>
  );
}

function ArticleRow({ point }) {
  return (
    <a className="article-row" href={point.url} target="_blank" rel="noreferrer">
      <span style={{ background: outletColor(point.outlet) }} />
      <div>
        <strong>{point.headline}</strong>
        <em>{point.outlet}</em>
      </div>
      <ExternalLink size={13} />
    </a>
  );
}
