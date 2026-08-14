window.__ModuleLoader__.load({
	id: "@dsh-external/dsh-engram-relay",
	factory: (require) => {
		var module = { exports: {} };
		var exports = module.exports;
		Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
		let react = require("react");
		let react_jsx_runtime = require("react/jsx-runtime");
		//#region src/client/force.ts
		const EPS = 1e-6;
		/**
		* 创建力导向模拟器（可持续迭代 + 增量加节点）。
		* nodes/edges/clusters 为**全量**引用（边按 id 存，body 增量加入后自动生效）；
		* 初始 nodes 非空时按簇散布摆位（全量一次算场景），否则空开始全走 addNode。
		*/
		function createForceSimulator(nodes, edges, opts) {
			const { width, height, charge = -300, spring = .1, springLength = 80, collideRadius = 24, centerStrength = .08, velocityDecay = .55, alphaDecay = .02, maxMove = 40, clusters, clusterTarget = 110, clusterStrength = .04, projectGroups, projectStrength = .03 } = opts;
			const cx = width / 2;
			const cy = height / 2;
			const ringRadius = Math.max(40, Math.min(width, height) / 2 - 60);
			const bodies = /* @__PURE__ */ new Map();
			const edgePairs = edges.filter((e) => e.from !== e.to).map((e) => ({
				from: e.from,
				to: e.to
			}));
			let alpha = 1;
			const jigOf = (i, salt = 0) => (i * 2654435761 + salt * 40503) % 1e3 / 1e3 - .5;
			if (nodes.length > 0) {
				if (projectGroups !== void 0 && projectGroups.size > 0) {
					const byProject = /* @__PURE__ */ new Map();
					for (const node of nodes) {
						const p = projectGroups.get(node.id) ?? "__solo__";
						const arr = byProject.get(p) ?? [];
						arr.push(node.id);
						byProject.set(p, arr);
					}
					const projectIds = [...byProject.keys()];
					const regionOf = /* @__PURE__ */ new Map();
					const sizes2 = projectIds.map((pid) => byProject.get(pid).length);
					const order2 = projectIds.map((_, i) => i).sort((a, b) => sizes2[b] - sizes2[a]);
					order2.forEach((oi, gi) => {
						const pid = projectIds[oi];
						const big = gi === 0;
						regionOf.set(pid, big ? {
							x: width / 2,
							y: height * .2
						} : {
							x: (gi - 1 + .5) / (order2.length - 1) * width,
							y: height * .8
						});
					});
					projectIds.forEach((pid) => {
						const members = byProject.get(pid);
						const center = regionOf.get(pid) ?? {
							x: cx,
							y: cy
						};
						members.forEach((id, mi) => {
							const inner = 2 * Math.PI * mi / Math.max(1, members.length);
							const rr = Math.min(90, 30 + mi % 6 * 10);
							const jig = jigOf(mi, members.length);
							const w = nodes.find((nd) => nd.id === id)?.weight ?? 1;
							bodies.set(id, {
								x: center.x + (rr + jig * 4) * Math.cos(inner) + jig * 4,
								y: center.y + (rr + jig * 4) * Math.sin(inner) + jig * 4,
								vx: 0,
								vy: 0,
								weight: Math.max(.5, w),
								r: nodes.find((nd) => nd.id === id)?.radius ?? 12
							});
						});
					});
				} else if (clusters !== void 0 && clusters.size > 0) {
					const byCluster = /* @__PURE__ */ new Map();
					for (const node of nodes) {
						const c = clusters.get(node.id) ?? "__solo__";
						const arr = byCluster.get(c) ?? [];
						arr.push(node.id);
						byCluster.set(c, arr);
					}
					const clusterIds = [...byCluster.keys()];
					const clusterCount = clusterIds.length;
					clusterIds.forEach((cid, ci) => {
						const angle = 2 * Math.PI * ci / clusterCount;
						const ccx = cx + ringRadius * Math.cos(angle);
						const ccy = cy + ringRadius * Math.sin(angle);
						const members = byCluster.get(cid);
						members.forEach((id, mi) => {
							const inner = clusterCount === 1 ? 0 : 2 * Math.PI * mi / Math.max(1, members.length);
							const rr = Math.min(70, 26 + mi % 5 * 8);
							const jig = jigOf(mi);
							const w = nodes.find((nd) => nd.id === id)?.weight ?? 1;
							bodies.set(id, {
								x: ccx + (rr + jig * 3) * Math.cos(inner) + jig * 3,
								y: ccy + (rr + jig * 3) * Math.sin(inner) + jig * 3,
								vx: 0,
								vy: 0,
								weight: Math.max(.5, w),
								r: nodes.find((nd) => nd.id === id)?.radius ?? 12
							});
						});
					});
				} else {
					const n = nodes.length;
					nodes.forEach((node, i) => {
						const angle = 2 * Math.PI * i / n;
						const jig = jigOf(i);
						bodies.set(node.id, {
							x: cx + (ringRadius + jig * 4) * Math.cos(angle) + jig * 4,
							y: cy + (ringRadius + jig * 4) * Math.sin(angle) + jig * 4,
							vx: 0,
							vy: 0,
							weight: Math.max(.5, node.weight ?? 1),
							r: node.radius ?? 12
						});
					});
				}
			}
			const projectRegion = /* @__PURE__ */ new Map();
			if (projectGroups !== void 0 && projectGroups.size > 0) {
				const groupIds = [...new Set(projectGroups.values())];
				const sizes = groupIds.map((gid) => [...projectGroups.values()].filter((v) => v === gid).length);
				const order = groupIds.map((_, i) => i).sort((a, b) => sizes[b] - sizes[a]);
				order.forEach((oi, gi) => {
					const gid = groupIds[oi];
					const big = gi === 0;
					if (gid === "__solo__") projectRegion.set(gid, {
						x: width * .85,
						y: height * .5
					});
					else if (big) projectRegion.set(gid, {
						x: width / 2,
						y: height * .2
					});
					else {
						const cw = width / (order.length - 2);
						projectRegion.set(gid, {
							x: (gi - 1 + .5) * cw,
							y: height * .8
						});
					}
				});
			}
			/** 单轮迭代（所有力 + 积分）。 */
			const iterate = (a) => {
				const list = [...bodies.entries()];
				for (let i = 0; i < list.length; i += 1) {
					const [, bi] = list[i];
					for (let j = i + 1; j < list.length; j += 1) {
						const [, bj] = list[j];
						const x = bj.x - bi.x;
						const y = bj.y - bi.y;
						let l = x * x + y * y;
						if (l < EPS) l = EPS;
						const w = charge * bi.weight * bj.weight * a / l;
						bi.vx += x * w;
						bi.vy += y * w;
						bj.vx -= x * w;
						bj.vy -= y * w;
					}
				}
				for (const { from, to } of edgePairs) {
					const s = bodies.get(from);
					const t = bodies.get(to);
					if (s === void 0 || t === void 0) continue;
					const x = t.x - s.x;
					const y = t.y - s.y;
					let l = Math.sqrt(x * x + y * y);
					if (l < EPS) l = EPS;
					l = (l - springLength) / l * a * spring;
					s.vx += x * l;
					s.vy += y * l;
					t.vx -= x * l;
					t.vy -= y * l;
				}
				if (clusters !== void 0 && clusters.size > 1 && list.length > 1) for (let i = 0; i < list.length; i += 1) {
					const [ida, bi] = list[i];
					const ca = clusters.get(ida);
					if (ca === void 0) continue;
					for (let j = i + 1; j < list.length; j += 1) {
						const [idb, bj] = list[j];
						if (clusters.get(idb) !== ca) continue;
						const x = bj.x - bi.x;
						const y = bj.y - bi.y;
						const dist = Math.sqrt(x * x + y * y);
						const dx = dist - clusterTarget;
						if (dx > 0) {
							const w = dx / dist * a * clusterStrength;
							bi.vx += x * w;
							bi.vy += y * w;
							bj.vx -= x * w;
							bj.vy -= y * w;
						}
					}
				}
				if (projectGroups !== void 0 && projectGroups.size > 0) {
					const pa = Math.max(a, .3);
					for (let i = 0; i < list.length; i += 1) {
						const [ida, bi] = list[i];
						const gid = projectGroups.get(ida);
						if (gid === void 0) continue;
						const c = projectRegion.get(gid);
						if (c === void 0) continue;
						bi.vx += (c.x - bi.x) * projectStrength * 2 * pa;
						bi.vy += (c.y - bi.y) * projectStrength * 2 * pa;
					}
				}
				for (let i = 0; i < list.length; i += 1) {
					const [, bi] = list[i];
					for (let j = i + 1; j < list.length; j += 1) {
						const [, bj] = list[j];
						const x = bj.x - bi.x;
						const y = bj.y - bi.y;
						let l = Math.sqrt(x * x + y * y);
						const r = (bi.r ?? collideRadius) + (bj.r ?? collideRadius);
						if (l < r) {
							if (l < EPS) {
								const ang = (i * 2654435761 + j * 40503) % 628 / 100;
								bj.x += Math.cos(ang) * .5;
								bj.y += Math.sin(ang) * .5;
								l = .5;
							}
							const overlap = (r - l) * .5;
							const nx = x / l;
							const ny = y / l;
							bi.x -= nx * overlap;
							bi.y -= ny * overlap;
							bj.x += nx * overlap;
							bj.y += ny * overlap;
							const push = (r - l) / l;
							bi.vx -= x * push * .5;
							bi.vy -= y * push * .5;
							bj.vx += x * push * .5;
							bj.vy += y * push * .5;
						}
					}
				}
				for (const body of bodies.values()) {
					body.vx += (cx - body.x) * centerStrength * a;
					body.vy += (cy - body.y) * centerStrength * a;
				}
				let sx = 0;
				let sy = 0;
				for (const body of bodies.values()) {
					sx += body.x;
					sy += body.y;
					body.vx *= velocityDecay;
					body.vy *= velocityDecay;
					const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy);
					if (speed > maxMove) {
						body.vx = body.vx / speed * maxMove;
						body.vy = body.vy / speed * maxMove;
					}
					body.x += body.vx;
					body.y += body.vy;
				}
				if (bodies.size > 0) {
					const mx = sx / bodies.size;
					for (const body of bodies.values()) body.x += cx - mx;
				}
			};
			return {
				step(iterations) {
					for (let iter = 0; iter < iterations; iter += 1) {
						alpha += (0 - alpha) * alphaDecay;
						iterate(alpha);
					}
				},
				addNode(id, weight = 1) {
					if (bodies.has(id)) return;
					const jig = jigOf(bodies.size, 7);
					bodies.set(id, {
						x: cx + jig * 24,
						y: cy + jigOf(bodies.size, 13) * 24,
						vx: 0,
						vy: 0,
						weight: Math.max(.5, weight),
						r: 12
					});
					alpha = Math.max(alpha, .6);
				},
				layout() {
					const blist = [...bodies.entries()];
					for (let pass = 0; pass < 8; pass += 1) {
						let changed = false;
						for (let i = 0; i < blist.length; i += 1) {
							const [, bi] = blist[i];
							for (let j = i + 1; j < blist.length; j += 1) {
								const [, bj] = blist[j];
								const dx = bj.x - bi.x;
								const dy = bj.y - bi.y;
								let d = Math.sqrt(dx * dx + dy * dy);
								const rr = (bi.r ?? 12) + (bj.r ?? 12);
								if (d < rr) {
									if (d < 1e-6) {
										const ang = (i * 2654435761 + j * 40503) % 628 / 100;
										bj.x += Math.cos(ang);
										bj.y += Math.sin(ang);
										d = 1;
									}
									const overlap = (rr - d) / 2;
									const nx = dx / d;
									const ny = dy / d;
									bi.x -= nx * overlap;
									bi.y -= ny * overlap;
									bj.x += nx * overlap;
									bj.y += ny * overlap;
									changed = true;
								}
							}
						}
						if (!changed) break;
					}
					const out = /* @__PURE__ */ new Map();
					for (const [id, body] of bodies) out.set(id, {
						x: body.x,
						y: body.y
					});
					return out;
				},
				alpha() {
					return alpha;
				}
			};
		}
		/** 一次性布局（兼容旧 API）：创建模拟器 + 全量迭代。 */
		function layoutForce(nodes, edges, opts) {
			if (nodes.length === 0) return /* @__PURE__ */ new Map();
			const sim = createForceSimulator(nodes, edges, opts);
			sim.step(opts.iterations ?? 500);
			return sim.layout();
		}
		//#endregion
		//#region \0dsh-css:D:\yjh\dsh\dsh-engram-relay\src\client\graph.module.css.mjs
		const css = ".w3nXOG_root{box-sizing:border-box;flex-direction:column;gap:8px;height:100%;max-height:100vh;padding:12px;font-size:13px;display:flex;overflow:hidden}.w3nXOG_toolbar{flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;display:flex}.w3nXOG_filters{gap:6px;display:flex}.w3nXOG_filterBtn{color:var(--ds-color-text-1,#d6d9de);cursor:pointer;background:#7f7f7f1f;border:1px solid #7f7f7f33;border-radius:6px;padding:3px 10px;font-size:12px;transition:background .15s}.w3nXOG_filterBtn:hover{background:#7f7f7f38}.w3nXOG_filterActive{color:#9db8ff;background:#4a7dff38;border-color:#4a7dff80}.w3nXOG_meta{align-items:center;gap:10px;display:flex}.w3nXOG_count{color:var(--ds-color-text-2,#8a94a6);font-size:12px}.w3nXOG_refresh{color:var(--ds-color-text-1,#d6d9de);cursor:pointer;background:0 0;border:1px solid #7f7f7f4d;border-radius:6px;padding:3px 10px;font-size:12px}.w3nXOG_refresh:hover{background:#7f7f7f2e}.w3nXOG_state{min-height:200px;color:var(--ds-color-text-2,#8a94a6);justify-content:center;align-items:center;display:flex}.w3nXOG_error{color:#ff6b6b;background:#ff6b6b14;border-radius:6px;padding:4px 8px;font-size:12px}.w3nXOG_canvas{background:#00000026;border:1px solid #7f7f7f26;border-radius:8px;flex:1;min-height:0;position:relative;overflow:hidden}.w3nXOG_svg{cursor:grab;touch-action:none;width:100%;height:100%;display:block;position:absolute;inset:0}.w3nXOG_edge,.w3nXOG_cluster{pointer-events:none}.w3nXOG_clusterLabel{fill:var(--ds-color-text-3,#6a7284);letter-spacing:.5px;user-select:none;font-size:11px;font-weight:600}.w3nXOG_node{cursor:pointer;outline:none}.w3nXOG_node:focus{outline:none}.w3nXOG_node circle{transition:r .15s,filter .15s}.w3nXOG_nodeSemantic circle{filter:drop-shadow(0 0 6px #4a7dff59)}.w3nXOG_nodeLabel{fill:var(--ds-color-text-2,#aab2c0);pointer-events:none;user-select:none;font-size:10px}.w3nXOG_nodeLabelSemantic{fill:var(--ds-color-text-1,#e8eaee);pointer-events:none;user-select:none;font-size:11px;font-weight:600}.w3nXOG_legend{color:var(--ds-color-text-2,#8a94a6);background:#00000059;border-radius:6px;align-items:center;gap:12px;padding:4px 10px;font-size:11px;display:flex;position:absolute;bottom:8px;left:10px}.w3nXOG_legendLineSolid{vertical-align:middle;border-top:2px solid #8a94a6;width:18px;height:0;margin-right:5px;display:inline-block}.w3nXOG_legendLineDash{vertical-align:middle;border-top:1px dashed #aab2c0;width:18px;height:0;margin-right:5px;display:inline-block}.w3nXOG_legendDot{vertical-align:middle;border-radius:50%;width:8px;height:8px;margin-right:5px;display:inline-block}.w3nXOG_legendCluster{vertical-align:middle;border:1.5px dashed #8a94a680;border-radius:50%;width:10px;height:10px;margin-right:5px;display:inline-block}.w3nXOG_detail{z-index:10;background:#0a0e16eb;border:1px solid #7f7f7f33;border-radius:8px;width:280px;padding:10px 12px;position:absolute;top:10px;bottom:10px;right:10px;overflow:auto;box-shadow:0 4px 20px #0006}.w3nXOG_detailHead{justify-content:space-between;align-items:center;gap:8px;display:flex}.w3nXOG_detailTitle{color:var(--ds-color-text-1,#e8eaee);font-weight:600}.w3nXOG_detailClose{color:var(--ds-color-text-2,#8a94a6);cursor:pointer;background:0 0;border:none;font-size:12px}.w3nXOG_detailSummary{color:var(--ds-color-text-2,#aab2c0);margin:6px 0}.w3nXOG_detailContent{white-space:pre-wrap;word-break:break-word;color:var(--ds-color-text-1,#d6d9de);background:#7f7f7f14;border-radius:6px;max-height:140px;margin:6px 0;padding:8px;font-size:12px;overflow:auto}.w3nXOG_detailSection{margin-top:8px}.w3nXOG_detailSectionTitle{color:var(--ds-color-text-2,#8a94a6);margin-bottom:4px;font-size:11px;font-weight:600;display:block}.w3nXOG_detailNode{color:#9db8ff;cursor:pointer;background:#4a7dff1f;border:1px solid #4a7dff4d;border-radius:4px;margin:0 6px 4px 0;padding:1px 8px;font-size:12px}.w3nXOG_detailNode:hover{background:#4a7dff38}.w3nXOG_detailNone{color:var(--ds-color-text-3,#5c6572);font-size:12px}";
		const tagId = "@dsh-external/dsh-engram-relay/graph.module.css";
		if (typeof document !== "undefined" && document.querySelector("style[data-plugin-css=" + JSON.stringify(tagId) + "]") === null) {
			const tag = document.createElement("style");
			tag.dataset.plugin = "@dsh-external/dsh-engram-relay";
			tag.dataset.pluginCss = tagId;
			tag.textContent = css;
			document.head.appendChild(tag);
		}
		var graph_module_css_default = {
			"cluster": "w3nXOG_cluster",
			"detailSectionTitle": "w3nXOG_detailSectionTitle",
			"filters": "w3nXOG_filters",
			"nodeLabel": "w3nXOG_nodeLabel",
			"node": "w3nXOG_node",
			"edge": "w3nXOG_edge",
			"legend": "w3nXOG_legend",
			"filterBtn": "w3nXOG_filterBtn",
			"detailSection": "w3nXOG_detailSection",
			"detailHead": "w3nXOG_detailHead",
			"legendDot": "w3nXOG_legendDot",
			"error": "w3nXOG_error",
			"root": "w3nXOG_root",
			"count": "w3nXOG_count",
			"svg": "w3nXOG_svg",
			"detailNone": "w3nXOG_detailNone",
			"nodeSemantic": "w3nXOG_nodeSemantic",
			"detail": "w3nXOG_detail",
			"detailTitle": "w3nXOG_detailTitle",
			"meta": "w3nXOG_meta",
			"toolbar": "w3nXOG_toolbar",
			"refresh": "w3nXOG_refresh",
			"detailClose": "w3nXOG_detailClose",
			"detailContent": "w3nXOG_detailContent",
			"filterActive": "w3nXOG_filterActive",
			"detailSummary": "w3nXOG_detailSummary",
			"legendLineDash": "w3nXOG_legendLineDash",
			"legendLineSolid": "w3nXOG_legendLineSolid",
			"legendCluster": "w3nXOG_legendCluster",
			"state": "w3nXOG_state",
			"canvas": "w3nXOG_canvas",
			"clusterLabel": "w3nXOG_clusterLabel",
			"detailNode": "w3nXOG_detailNode",
			"nodeLabelSemantic": "w3nXOG_nodeLabelSemantic"
		};
		//#endregion
		//#region src/client/GraphView.tsx
		/**
		* GraphView — 记忆图谱可视化（DSH 会话页「图谱」Tab）。
		*
		* 数据面：host 的 /engram-relay/api/graph（分层准入：global + 本目录
		* project + 本会话 session）。渲染：确定性力导向布局（force.ts）+ SVG。
		*  - 节点 = 记忆（颜色按层：global 蓝 / project 绿 / session 橙）
		*  - 实线边 = 因果（causes），虚线边 = 双向链接（link）
		*  - 点击节点 → 拉取详情（渐进披露第二层：正文/前因/后果/关联）
		*  - 层过滤 + 刷新
		*/
		const VIEW_W = 900;
		const VIEW_H = 620;
		function GraphView({ t, sessionId }) {
			const [data, setData] = (0, react.useState)(null);
			const [loading, setLoading] = (0, react.useState)(true);
			const [error, setError] = (0, react.useState)("");
			const [view, setView] = (0, react.useState)({
				vx: 0,
				vy: 0,
				vw: VIEW_W,
				vh: VIEW_H
			});
			const viewRef = (0, react.useRef)(view);
			viewRef.current = view;
			const svgRef = (0, react.useRef)(null);
			const dragRef = (0, react.useRef)(null);
			const lastWheelRef = (0, react.useRef)(0);
			const [canvasSize, setCanvasSize] = (0, react.useState)({
				w: VIEW_W,
				h: VIEW_H
			});
			(0, react.useEffect)(() => {
				const el = svgRef.current?.parentElement;
				if (!el) return;
				const ro = new ResizeObserver(() => {
					const r = el.getBoundingClientRect();
					if (r.width > 0 && r.height > 0) setCanvasSize({
						w: r.width,
						h: r.height
					});
				});
				ro.observe(el);
				return () => ro.disconnect();
			}, [loading]);
			const zoomScale = VIEW_W / view.vw;
			const zc = Math.max(1, Math.min(zoomScale, 2.5));
			const loadGraph = () => {
				setLoading(true);
				setError("");
				const q = sessionId !== void 0 && sessionId !== "" ? `?sessionId=${encodeURIComponent(sessionId)}` : "";
				fetch(`/engram-relay/api/graph${q}`).then((res) => res.ok ? res.json() : Promise.reject(/* @__PURE__ */ new Error(`HTTP ${res.status}`))).then((d) => setData(d)).catch((e) => setError(String(e?.message ?? e))).finally(() => setLoading(false));
			};
			(0, react.useEffect)(() => {
				loadGraph();
			}, [sessionId]);
			const { nodes, edges } = (0, react.useMemo)(() => {
				if (data === null) return {
					nodes: [],
					edges: []
				};
				const ids = new Set(data.nodes.map((n) => n.id));
				const visibleEdges = data.edges.filter((e) => ids.has(e.from) && ids.has(e.to));
				return {
					nodes: data.nodes,
					edges: visibleEdges
				};
			}, [data]);
			const clusters = (0, react.useMemo)(() => {
				if (nodes.length === 0) return {
					list: [],
					clusterOf: /* @__PURE__ */ new Map()
				};
				const adj = /* @__PURE__ */ new Map();
				for (const n of nodes) adj.set(n.id, /* @__PURE__ */ new Set());
				for (const e of edges) {
					adj.get(e.from)?.add(e.to);
					adj.get(e.to)?.add(e.from);
				}
				const visited = /* @__PURE__ */ new Set();
				const list = [];
				const clusterOf = /* @__PURE__ */ new Map();
				for (const n of nodes) {
					if (visited.has(n.id)) continue;
					const ids = [];
					const projects = /* @__PURE__ */ new Set();
					const queue = [n.id];
					visited.add(n.id);
					while (queue.length > 0) {
						const id = queue.shift();
						ids.push(id);
						projects.add(nodes.find((x) => x.id === id)?.projectId ?? null);
						for (const nb of adj.get(id) ?? []) if (!visited.has(nb)) {
							visited.add(nb);
							queue.push(nb);
						}
					}
					const cid = `c${list.length}`;
					for (const id of ids) clusterOf.set(id, cid);
					list.push({
						ids,
						projects
					});
				}
				return {
					list,
					clusterOf
				};
			}, [nodes, edges]);
			clusters.list;
			const clusterOf = clusters.clusterOf;
			const degreeOf = (0, react.useMemo)(() => {
				const deg = /* @__PURE__ */ new Map();
				for (const e of edges) {
					deg.set(e.from, (deg.get(e.from) ?? 0) + 1);
					deg.set(e.to, (deg.get(e.to) ?? 0) + 1);
				}
				return deg;
			}, [edges]);
			const layout = (0, react.useMemo)(() => {
				const projectGroups = /* @__PURE__ */ new Map();
				for (const n of nodes) projectGroups.set(n.id, n.projectId ?? "__solo__");
				return layoutForce(nodes.map((n) => {
					const deg = degreeOf.get(n.id) ?? 0;
					const isSem = n.state === "semantic";
					const isEvt = n.kind === "event";
					const rBase = (7 + Math.min(9, deg * 1.2) + (isSem ? 3 : 0) + n.importance * 1.5) * (isEvt ? .7 : 1);
					return {
						id: n.id,
						weight: .6 + n.importance,
						radius: Math.max(12, rBase)
					};
				}), edges.map((e) => ({
					from: e.from,
					to: e.to
				})), {
					width: canvasSize.w,
					height: canvasSize.h,
					iterations: 250,
					charge: -100,
					spring: .1,
					springLength: 110,
					collideRadius: 22,
					centerStrength: .08,
					clusters: clusterOf.size > 0 ? clusterOf : void 0,
					clusterTarget: 110,
					clusterStrength: .04,
					projectGroups: projectGroups.size > 0 ? projectGroups : void 0,
					projectStrength: .8
				});
			}, [
				nodes,
				edges,
				clusterOf,
				canvasSize
			]);
			const cull = (p, margin = 150 / zc) => p.x >= view.vx - margin && p.x <= view.vx + view.vw + margin && p.y >= view.vy - margin && p.y <= view.vy + view.vh + margin;
			const showLabels = zoomScale >= 1;
			const [selectedId, setSelectedId] = (0, react.useState)(null);
			const highlight = (0, react.useMemo)(() => {
				const sel = selectedId;
				if (sel === null) return null;
				const nodeIds = /* @__PURE__ */ new Set([sel]);
				const edgeKeys = /* @__PURE__ */ new Set();
				for (const e of edges) if (e.from === sel || e.to === sel) {
					nodeIds.add(e.from);
					nodeIds.add(e.to);
					edgeKeys.add(`${e.from}|${e.to}|${e.kind}`);
				}
				return {
					nodeIds,
					edgeKeys
				};
			}, [selectedId, edges]);
			const fitToCurrent = () => {
				const pts = nodes.map((n) => layout.get(n.id)).filter((p) => p !== void 0);
				if (pts.length === 0) return null;
				const pad = 80;
				const minX = pts.reduce((s, p) => Math.min(s, p.x), Infinity) - pad;
				const maxX = pts.reduce((s, p) => Math.max(s, p.x), -Infinity) + pad;
				const minY = pts.reduce((s, p) => Math.min(s, p.y), Infinity) - pad;
				const maxY = pts.reduce((s, p) => Math.max(s, p.y), -Infinity) + pad;
				return {
					vx: minX,
					vy: minY,
					vw: Math.max(200, maxX - minX),
					vh: Math.max(150, maxY - minY)
				};
			};
			const fitToCurrentRef = (0, react.useRef)(fitToCurrent);
			fitToCurrentRef.current = fitToCurrent;
			const lastFitKeyRef = (0, react.useRef)("");
			(0, react.useEffect)(() => {
				if (nodes.length === 0 || layout.size === 0) return;
				const key = `${nodes.length}|${canvasSize.w}x${canvasSize.h}`;
				if (lastFitKeyRef.current === key) return;
				lastFitKeyRef.current = key;
				const f = fitToCurrent();
				if (f !== null) setView(f);
			}, [
				nodes,
				layout,
				canvasSize
			]);
			try {
				const byProject = /* @__PURE__ */ new Map();
				for (const n of nodes) {
					const arr = byProject.get(n.projectId) ?? [];
					arr.push(n.id);
					byProject.set(n.projectId, arr);
				}
				const centers = [];
				for (const [pid, ids] of byProject) {
					if (pid === null || ids.length < 3) continue;
					const pts = ids.map((id) => layout.get(id)).filter((p) => p !== void 0);
					if (pts.length < 3) continue;
					centers.push({
						cx: pts.reduce((s, p) => s + p.x, 0) / pts.length,
						cy: pts.reduce((s, p) => s + p.y, 0) / pts.length,
						label: String(pid).split(/[\\/]/).pop() || "项目"
					});
				}
				const overlaps = [];
				for (let i = 0; i < centers.length; i += 1) for (let j = i + 1; j < centers.length; j += 1) {
					const a = centers[i], b = centers[j];
					const d = Math.hypot(a.cx - b.cx, a.cy - b.cy);
					if (d < 220) overlaps.push(`[${a.label}]↔[${b.label}] 质心距${d.toFixed(0)}`);
				}
				if (overlaps.length > 0) {
					const line = `[${(/* @__PURE__ */ new Date()).toISOString()}] 聚团过近 ${overlaps.length} 对: ${overlaps.join("; ")}\n`;
					fetch("/engram-relay/api/graph-log", {
						method: "POST",
						headers: { "Content-Type": "text/plain" },
						body: line
					});
				}
			} catch {}
			/** 项目着色（v0.4）：projectId 哈希取色；null（通用知识）灰色。
			*  ⚠️ function 声明（提升）——projectCircles useMemo 在其定义前引用，
			*  const 箭头函数会 TDZ 报错（Cannot access before initialization）。 */
			function projectColor(projectId) {
				if (projectId === null) return "#8a94a6";
				let h = 0;
				for (const ch of projectId) h = h * 31 + ch.charCodeAt(0) >>> 0;
				return `hsl(${h % 360} 55% 55%)`;
			}
			/** 簇着色（v0.5 取经 Obsidian color groups：**同簇同色**——分类感知的
			*  核心。簇色相从簇 id 确定性哈希；孤立节点回退项目色）。 */
			function clusterColor(clusterId) {
				let h = 0;
				for (const ch of clusterId) h = h * 131 + ch.charCodeAt(0) >>> 0;
				return `hsl(${(h + 35) % 360} 65% 62%)`;
			}
			(0, react.useEffect)(() => {
				const svg = svgRef.current;
				if (svg === null) return;
				const onWheel = (e) => {
					if (!e.ctrlKey) return;
					e.preventDefault();
					const now = performance.now();
					if (now - lastWheelRef.current < 50) return;
					lastWheelRef.current = now;
					const rect = svg.getBoundingClientRect();
					const mx = (e.clientX - rect.left) / rect.width;
					const my = (e.clientY - rect.top) / rect.height;
					const factor = e.deltaY < 0 ? 1 / 1.15 : 1.15;
					setView((v) => {
						const vw2 = Math.min(1e5, Math.max(10, v.vw * factor));
						const vh2 = v.vh * factor;
						const wx = v.vx + mx * v.vw;
						const wy = v.vy + my * v.vh;
						return {
							vx: wx - mx * vw2,
							vy: wy - my * vh2,
							vw: vw2,
							vh: vh2
						};
					});
				};
				svg.addEventListener("wheel", onWheel, { passive: false });
				return () => svg.removeEventListener("wheel", onWheel);
			}, [loading, nodes.length]);
			(0, react.useEffect)(() => {
				const svg = svgRef.current;
				if (svg === null) return;
				const onDown = (e) => {
					if (e.button !== 0) return;
					if (e.target.closest?.("[data-node-id]")) return;
					setSelectedId(null);
					const rect = svg.getBoundingClientRect();
					dragRef.current = {
						startX: e.clientX,
						startY: e.clientY,
						startView: viewRef.current,
						rect,
						moved: false
					};
					e.preventDefault();
				};
				const onMove = (e) => {
					const d = dragRef.current;
					if (d === null) return;
					d.moved = d.moved || Math.hypot(e.clientX - d.startX, e.clientY - d.startY) > 3;
					const dx = (e.clientX - d.startX) / d.rect.width;
					const dy = (e.clientY - d.startY) / d.rect.height;
					setView((v) => ({
						...v,
						vx: d.startView.vx - dx * d.startView.vw,
						vy: d.startView.vy - dy * d.startView.vh
					}));
				};
				const onUp = () => {
					const d = dragRef.current;
					dragRef.current = null;
					if (d !== null && !d.moved) setSelectedId(null);
				};
				svg.addEventListener("mousedown", onDown);
				window.addEventListener("mousemove", onMove);
				window.addEventListener("mouseup", onUp);
				return () => {
					svg.removeEventListener("mousedown", onDown);
					window.removeEventListener("mousemove", onMove);
					window.removeEventListener("mouseup", onUp);
				};
			}, [loading, nodes.length]);
			return /* @__PURE__ */ (0, react_jsx_runtime.jsxs)("div", {
				className: graph_module_css_default.root,
				children: [
					/* @__PURE__ */ (0, react_jsx_runtime.jsx)("div", {
						className: graph_module_css_default.toolbar,
						children: /* @__PURE__ */ (0, react_jsx_runtime.jsxs)("div", {
							className: graph_module_css_default.meta,
							children: [
								/* @__PURE__ */ (0, react_jsx_runtime.jsx)("span", {
									className: graph_module_css_default.count,
									children: data !== null ? t("graph.count", {
										nodes: nodes.length,
										edges: edges.length
									}) : ""
								}),
								/* @__PURE__ */ (0, react_jsx_runtime.jsx)("button", {
									className: graph_module_css_default.refresh,
									onClick: () => {
										const f = fitToCurrent();
										if (f !== null) setView(f);
									},
									children: t("graph.reset")
								}),
								/* @__PURE__ */ (0, react_jsx_runtime.jsx)("button", {
									className: graph_module_css_default.refresh,
									onClick: loadGraph,
									children: t("graph.refresh")
								})
							]
						})
					}),
					error !== "" && /* @__PURE__ */ (0, react_jsx_runtime.jsx)("div", {
						className: graph_module_css_default.error,
						children: error
					}),
					loading && /* @__PURE__ */ (0, react_jsx_runtime.jsx)("div", {
						className: graph_module_css_default.state,
						children: t("graph.loading")
					}),
					!loading && data !== null && nodes.length === 0 && /* @__PURE__ */ (0, react_jsx_runtime.jsx)("div", {
						className: graph_module_css_default.state,
						children: t("graph.empty")
					}),
					!loading && nodes.length > 0 && /* @__PURE__ */ (0, react_jsx_runtime.jsxs)("div", {
						className: graph_module_css_default.canvas,
						children: [/* @__PURE__ */ (0, react_jsx_runtime.jsxs)("svg", {
							ref: svgRef,
							viewBox: `${view.vx} ${view.vy} ${view.vw} ${view.vh}`,
							className: graph_module_css_default.svg,
							children: [
								/* @__PURE__ */ (0, react_jsx_runtime.jsxs)("defs", { children: [/* @__PURE__ */ (0, react_jsx_runtime.jsx)("pattern", {
									id: "dot-grid",
									width: "40",
									height: "40",
									patternUnits: "userSpaceOnUse",
									children: /* @__PURE__ */ (0, react_jsx_runtime.jsx)("circle", {
										cx: "1.5",
										cy: "1.5",
										r: "1.2",
										fill: "rgba(255,255,255,0.06)"
									})
								}), /* @__PURE__ */ (0, react_jsx_runtime.jsxs)("radialGradient", {
									id: "halo-grad",
									children: [/* @__PURE__ */ (0, react_jsx_runtime.jsx)("stop", {
										offset: "0%",
										stopColor: "#4a7dff",
										stopOpacity: "0.28"
									}), /* @__PURE__ */ (0, react_jsx_runtime.jsx)("stop", {
										offset: "100%",
										stopColor: "#4a7dff",
										stopOpacity: "0"
									})]
								})] }),
								/* @__PURE__ */ (0, react_jsx_runtime.jsx)("rect", {
									x: view.vx - 2e3,
									y: view.vy - 2e3,
									width: view.vw + 4e3,
									height: view.vh + 4e3,
									fill: "url(#dot-grid)"
								}),
								edges.map((e) => {
									const a = layout.get(e.from);
									const b = layout.get(e.to);
									if (a === void 0 || b === void 0) return null;
									if (!cull(a) && !cull(b)) return null;
									const key = `${e.from}|${e.to}|${e.kind}`;
									const isHighlighted = highlight !== null && highlight.edgeKeys.has(key);
									const baseOpacity = highlight !== null && !isHighlighted ? .04 : isHighlighted ? .95 : .35;
									if (e.kind === "causes") {
										const src = nodes.find((n) => n.id === e.from);
										const scid = src ? clusterOf.get(src.id) : void 0;
										const color = scid !== void 0 ? clusterColor(scid) : src ? projectColor(src.projectId) : "#8a94a6";
										return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("line", {
											x1: a.x,
											y1: a.y,
											x2: b.x,
											y2: b.y,
											stroke: isHighlighted ? "#ffd166" : color,
											strokeOpacity: baseOpacity,
											strokeWidth: (isHighlighted ? 2.5 : 1.5) / zc,
											className: graph_module_css_default.edge
										}, key);
									}
									return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("line", {
										x1: a.x,
										y1: a.y,
										x2: b.x,
										y2: b.y,
										stroke: isHighlighted ? "#ffd166" : "#aab2c0",
										strokeOpacity: baseOpacity,
										strokeWidth: (isHighlighted ? 2 : 1) / zc,
										strokeDasharray: "5 4",
										className: graph_module_css_default.edge
									}, key);
								}),
								nodes.map((n) => {
									const p = layout.get(n.id);
									if (p === void 0) return null;
									if (!cull(p)) return null;
									const cid = clusterOf.get(n.id);
									const color = cid !== void 0 ? clusterColor(cid) : projectColor(n.projectId);
									const selected = selectedId === n.id;
									const isSemantic = n.state === "semantic";
									const isEvent = n.kind === "event";
									const deg = degreeOf.get(n.id) ?? 0;
									const r = ((7 + Math.min(9, deg * 1.2) + (isSemantic ? 3 : 0) + n.importance * 1.5) * (isEvent ? .7 : 1) + (selected ? 2 : 0)) / zc;
									const inHighlight = highlight !== null && highlight.nodeIds.has(n.id);
									const opacity = (highlight !== null && !inHighlight ? .12 : 1) * (isEvent ? .55 : 1);
									return /* @__PURE__ */ (0, react_jsx_runtime.jsxs)("g", {
										"data-node-id": n.id,
										className: `${graph_module_css_default.node} ${isSemantic ? graph_module_css_default.nodeSemantic : ""}`,
										onClick: () => setSelectedId(n.id),
										role: "button",
										tabIndex: 0,
										opacity,
										children: [
											/* @__PURE__ */ (0, react_jsx_runtime.jsx)("circle", {
												cx: p.x,
												cy: p.y,
												r: r + 22 / zc,
												fill: "transparent",
												pointerEvents: "all"
											}),
											isSemantic && /* @__PURE__ */ (0, react_jsx_runtime.jsx)("circle", {
												cx: p.x,
												cy: p.y,
												r: r + 9 / zc,
												fill: "url(#halo-grad)",
												pointerEvents: "none"
											}),
											/* @__PURE__ */ (0, react_jsx_runtime.jsx)("circle", {
												cx: p.x,
												cy: p.y,
												r,
												fill: color,
												fillOpacity: selected ? 1 : isSemantic ? .95 : .82,
												stroke: selected ? "#fff" : isSemantic ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.35)",
												strokeWidth: (selected ? 2 : isSemantic ? 1.5 : 1) / zc
											}),
											showLabels && /* @__PURE__ */ (0, react_jsx_runtime.jsx)("text", {
												x: p.x,
												y: p.y + r + 12 / zc,
												textAnchor: "middle",
												className: isSemantic ? graph_module_css_default.nodeLabelSemantic : graph_module_css_default.nodeLabel,
												style: {
													fontSize: (isSemantic ? 11 : 10) / zc,
													paintOrder: "stroke",
													stroke: "rgba(10,14,22,0.9)",
													strokeWidth: 3 / zc
												},
												children: n.title.length > 14 ? `${n.title.slice(0, 13)}…` : n.title
											})
										]
									}, n.id);
								})
							]
						}), /* @__PURE__ */ (0, react_jsx_runtime.jsxs)("div", {
							className: graph_module_css_default.legend,
							children: [
								/* @__PURE__ */ (0, react_jsx_runtime.jsxs)("span", { children: [/* @__PURE__ */ (0, react_jsx_runtime.jsx)("span", { className: graph_module_css_default.legendLineSolid }), t("graph.legend.causes")] }),
								/* @__PURE__ */ (0, react_jsx_runtime.jsxs)("span", { children: [/* @__PURE__ */ (0, react_jsx_runtime.jsx)("span", { className: graph_module_css_default.legendLineDash }), t("graph.legend.link")] }),
								/* @__PURE__ */ (0, react_jsx_runtime.jsxs)("span", { children: [/* @__PURE__ */ (0, react_jsx_runtime.jsx)("span", {
									className: graph_module_css_default.legendDot,
									style: { background: projectColor("D:\\x") }
								}), t("graph.legend.project")] }),
								/* @__PURE__ */ (0, react_jsx_runtime.jsxs)("span", { children: [/* @__PURE__ */ (0, react_jsx_runtime.jsx)("span", {
									className: graph_module_css_default.legendDot,
									style: { background: projectColor(null) }
								}), t("graph.legend.solo")] })
							]
						})]
					})
				]
			});
		}
		//#endregion
		//#region src/client/locales.ts
		/**
		* 图谱 Tab 词典（zh/en，key 集单一真源 = zh）。
		*/
		const zh = {
			"graphTab.label": "图谱",
			"graph.loading": "加载记忆图谱…",
			"graph.loadFailed": "加载失败：{message}",
			"graph.empty": "（暂无记忆——模型经 engram_store 写入后，图谱会在这里生长）",
			"graph.refresh": "刷新",
			"graph.reset": "重置视图",
			"graph.filter.all": "全部",
			"graph.filter.global": "通用",
			"graph.filter.project": "项目",
			"graph.filter.title": "显示过滤（仅影响显示——图谱始终是一张图，数据不隔离）",
			"graph.count": "{nodes} 节点 · {edges} 边",
			"graph.layer.global": "通用",
			"graph.layer.project": "项目",
			"graph.edge.causes": "因果",
			"graph.edge.link": "链接",
			"graph.detail.title": "节点详情",
			"graph.detail.kind": "类型",
			"graph.detail.layer": "层级",
			"graph.detail.content": "正文",
			"graph.detail.causes": "前因（因果 ↑）",
			"graph.detail.effects": "后果（因果 ↓）",
			"graph.detail.links": "关联（双向链接）",
			"graph.detail.none": "（无）",
			"graph.detail.close": "关闭",
			"graph.detail.open": "展开详情",
			"graph.legend.causes": "—— 因果边",
			"graph.legend.link": "··· 双向链接",
			"graph.legend.project": "● 项目（同色聚团）",
			"graph.legend.solo": "○ 通用知识",
			"graph.hint": "节点=记忆（**同项目同色聚团**，大小=连接数，发光=固化知识）；实线=因果，虚线=双向链接；**单击**节点高亮其延展边，**点空白**取消高亮/平移画布；Ctrl+滚轮缩放。"
		};
		const en = {
			"graphTab.label": "Graph",
			"graph.loading": "Loading memory graph…",
			"graph.loadFailed": "Failed to load: {message}",
			"graph.empty": "(No memories yet — once the model writes via engram_store, the graph grows here)",
			"graph.refresh": "Refresh",
			"graph.reset": "Reset view",
			"graph.filter.all": "All",
			"graph.filter.global": "General",
			"graph.filter.project": "Project",
			"graph.filter.title": "Display filter (view only — the graph is always one canvas, data is not isolated)",
			"graph.count": "{nodes} nodes · {edges} edges",
			"graph.layer.global": "General",
			"graph.layer.project": "Project",
			"graph.edge.causes": "causal",
			"graph.edge.link": "link",
			"graph.detail.title": "Node details",
			"graph.detail.kind": "Kind",
			"graph.detail.layer": "Layer",
			"graph.detail.content": "Content",
			"graph.detail.causes": "Causes (↑)",
			"graph.detail.effects": "Effects (↓)",
			"graph.detail.links": "Links",
			"graph.detail.none": "(none)",
			"graph.detail.close": "Close",
			"graph.detail.open": "Expand",
			"graph.legend.causes": "—— causal edge",
			"graph.legend.link": "··· bidirectional link",
			"graph.legend.project": "● project (same color clusters)",
			"graph.legend.solo": "○ general knowledge",
			"graph.hint": "Nodes are memories (same project clusters by color; size = link count; glow = consolidated knowledge); solid = causal, dashed = link; single-click a node to highlight its edges, click empty space to deselect/pan; Ctrl+wheel to zoom."
		};
		//#endregion
		//#region src/client/index.ts
		/**
		* dsh-engram-relay — client entry：注册会话页「图谱」Tab。
		*
		* 记忆图谱可视化：host 端 /engram-relay/api/graph 提供分层准入后的
		* 节点+边数据，本 Tab 用确定性力导向布局渲染 SVG（节点=记忆·颜色分层，
		* 实线=因果边，虚线=双向链接），点击节点展开详情（渐进披露第二层）。
		*
		* 装配：探测 graph API 存在（host 加载成功）才注册 Tab；label 绑定词典。
		*/
		/** 插件显示名（诊断用）。 */
		const name = "dsh-engram-relay-client";
		/** 依赖服务：客户端 slots 注册表 + locale 词典服务。 */
		const inject = ["slots", "locale"];
		/** 客户端文案命名空间。 */
		const LOCALE_NS = "engram.relay";
		function apply(ctx) {
			ctx.effect(() => ctx.locale.register(LOCALE_NS, {
				zh,
				en
			}), "dsh-engram-relay: dictionaries");
			const t = ctx.locale.bind(LOCALE_NS);
			let cancelled = false;
			let disposeTab;
			fetch("/engram-relay/api/graph").then((res) => res.ok ? res.json() : Promise.reject(/* @__PURE__ */ new Error(`HTTP ${res.status}`))).then(() => {
				if (cancelled) return;
				disposeTab = ctx.slots.inject("conversation.view", () => ctx.slots.register({
					name: "conversation.view",
					id: "engram-graph",
					order: 24,
					label: () => t("graphTab.label"),
					locale: LOCALE_NS
				}, (props) => {
					const sessionId = props.sessionId;
					return GraphView({
						t,
						sessionId
					});
				}));
			}).catch(() => {});
			ctx.effect(() => () => {
				cancelled = true;
				disposeTab?.();
			}, "dsh-engram-relay: graph tab");
		}
		//#endregion
		exports.LOCALE_NS = LOCALE_NS;
		exports.apply = apply;
		exports.inject = inject;
		exports.name = name;
		return module.exports;
	}
});

//# sourceMappingURL=client.js.map