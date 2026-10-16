import { describe, expect, it } from "vitest";
import { createMemoryNetwork3DLayout, layoutMemoryNetwork3D, memory3DVolumeRatio, memoryNetwork3DLayoutSnapshot, projectMemoryPoint, stepMemoryNetwork3DLayout, DEFAULT_MEMORY_CAMERA } from "../src/features/memory-workspace/memory-3d.js";
import { decodeMemoryGraphPickColor, MAX_MEMORY_GRAPH_LABELS, MAX_MEMORY_GRAPH_LABEL_MEASUREMENTS, memoryGraphP95, selectMemoryGraphLabels } from "../src/features/memory-workspace/memory-webgl-renderer.js";
import type { MemoryGraphNode, MemoryGraphEdge } from "@space/contracts";

describe("3D memory projection", () => {
  it("rotates all three axes through a full turn and returns to the same view", () => {
    const point = { x: 45, y: 20, z: 60 };
    const initial = projectMemoryPoint(point, DEFAULT_MEMORY_CAMERA, 2, 1000, 700);
    const rotated = projectMemoryPoint(point, { ...DEFAULT_MEMORY_CAMERA, yaw: DEFAULT_MEMORY_CAMERA.yaw + Math.PI / 2 }, 2, 1000, 700);
    const full = projectMemoryPoint(point, { ...DEFAULT_MEMORY_CAMERA, yaw: DEFAULT_MEMORY_CAMERA.yaw + Math.PI * 2, pitch: DEFAULT_MEMORY_CAMERA.pitch + Math.PI * 2 }, 2, 1000, 700);
    expect(rotated.x).not.toBeCloseTo(initial.x);
    expect(rotated.z).not.toBeCloseTo(initial.z);
    expect(full.x).toBeCloseTo(initial.x);
    expect(full.y).toBeCloseTo(initial.y);
    expect(full.z).toBeCloseTo(initial.z);
  });
  const fixture = () => {
    const nodes: MemoryGraphNode[] = [{ id: "root", label: "gemini.md", type: "SOURCE", sourcePath: "/opt/spaceapp/docs/gemini.md", recordId: null }];
    const edges: MemoryGraphEdge[] = [];
    for (let file = 0; file < 12; file++) {
      const id = `source-${file}`;
      nodes.push({ id, label: `${id}.md`, type: "SOURCE", sourcePath: `/opt/spaceapp/docs/${id}.md`, recordId: null });
      edges.push({ id: `file-${file}`, type: "DERIVED_FROM", source: "root", target: id });
      for (let item = 0; item < 30; item++) {
        const memory = `${id}-memory-${item}`;
        nodes.push({ id: memory, label: memory, type: "MEMORY", sourcePath: `/opt/spaceapp/docs/${id}.md`, recordId: memory });
        edges.push({ id: memory, type: "CONTAINS", source: id, target: memory });
      }
    }
    return { nodes, edges };
  };
  it("occupies a volume, including at a side-on camera angle", () => {
    const { nodes, edges } = fixture();
    const layout = layoutMemoryNetwork3D(nodes, edges);
    expect(layout.size).toBe(nodes.length);
    expect(layout.get("root")).toEqual({ x: 0, y: 0, z: 0 });
    expect(memory3DVolumeRatio([...layout.values()])).toBeGreaterThan(0.35);
    for (const yaw of [0, Math.PI / 2, Math.PI]) {
      const points = [...layout.values()].map(p => projectMemoryPoint(p, { yaw, pitch: 0, zoom: 1 }, 1, 1000, 700));
      const width = Math.max(...points.map(p => p.x)) - Math.min(...points.map(p => p.x));
      const height = Math.max(...points.map(p => p.y)) - Math.min(...points.map(p => p.y));
      expect(width / height).toBeGreaterThan(0.5);
      expect(width / height).toBeLessThan(2);
    }
  });
  it("keeps real source links close and is deterministic under input reordering", () => {
    const { nodes, edges } = fixture();
    const layout = layoutMemoryNetwork3D(nodes, edges);
    const reversed = layoutMemoryNetwork3D([...nodes].reverse(), [...edges].reverse());
    expect([...reversed]).toEqual([...layout]);
    const distance = (a: string, b: string) => { const p = layout.get(a)!, q = layout.get(b)!; return Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z); };
    const related = edges.filter(edge => edge.type === "CONTAINS");
    const meanLink = related.reduce((sum, edge) => sum + distance(edge.source, edge.target), 0) / related.length;
    const meanUnrelated = related.reduce((sum, edge) => sum + distance(edge.source === "source-0" ? "source-6" : "source-0", edge.target), 0) / related.length;
    expect(meanLink).toBeLessThan(meanUnrelated * 0.7);
  });
  it("detects flat geometry even if the plane is tilted", () => {
    const plane = Array.from({ length: 100 }, (_, i) => ({ x: i % 10, y: Math.floor(i / 10), z: i % 10 + Math.floor(i / 10) }));
    expect(memory3DVolumeRatio(plane)).toBeCloseTo(0);
    expect(layoutMemoryNetwork3D([], []).size).toBe(0);
  });
  it("streams deterministic layout snapshots before converging", () => {
    const { nodes, edges } = fixture();
    const state = createMemoryNetwork3DLayout(nodes, edges);
    const initial = memoryNetwork3DLayoutSnapshot(state);
    expect(stepMemoryNetwork3DLayout(state, 12)).toBe(false);
    expect(memoryNetwork3DLayoutSnapshot(state)).not.toEqual(initial);
    while (!stepMemoryNetwork3DLayout(state, 12)) { /* worker-sized chunks */ }
    expect(memoryNetwork3DLayoutSnapshot(state)).toEqual(layoutMemoryNetwork3D(nodes, edges));
  });
  it("bounds label measurement and drawing independently of graph size", () => {
    const nodes: MemoryGraphNode[] = Array.from({ length: 5000 }, (_, index) => ({
      id: `source:${index}`,
      label: index === 0 ? "gemini.md" : `Source ${index}`,
      type: "SOURCE",
      sourcePath: `/memory/${index}.md`,
      recordId: null
    }));
    const coordinates = new Map(nodes.map((node, index) => [node.id, { x: (index % 50) - 25, y: Math.floor(index / 50) - 50, z: index % 7 }]));
    let measurements = 0;
    const labels = selectMemoryGraphLabels(nodes, coordinates, DEFAULT_MEMORY_CAMERA, 3, 1200, 800, "source:4999", label => {
      measurements++;
      return label.length * 6;
    });
    expect(measurements).toBeLessThanOrEqual(MAX_MEMORY_GRAPH_LABEL_MEASUREMENTS);
    expect(labels.length).toBeLessThanOrEqual(MAX_MEMORY_GRAPH_LABELS);
    expect(labels.some(label => label.id === "source:4999")).toBe(true);
  });
  it("decodes GPU picking colors and reports the nearest-rank p95", () => {
    expect(decodeMemoryGraphPickColor([0, 0, 0, 0])).toBeNull();
    expect(decodeMemoryGraphPickColor([1, 0, 0, 255])).toBe(0);
    expect(decodeMemoryGraphPickColor([0, 1, 0, 255])).toBe(255);
    expect(memoryGraphP95(Array.from({ length: 100 }, (_, index) => index + 1))).toBe(95);
    expect(memoryGraphP95([])).toBe(0);
  });
});
