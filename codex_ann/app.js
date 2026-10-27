const canvas = document.querySelector("#ann-canvas");
const ctx = canvas.getContext("2d");

const els = {
  tabs: [...document.querySelectorAll(".algo-tab")],
  title: document.querySelector("#algo-title"),
  description: document.querySelector("#algo-description"),
  stepCounter: document.querySelector("#step-counter"),
  prev: document.querySelector("#prev-step"),
  next: document.querySelector("#next-step"),
  play: document.querySelector("#play-toggle"),
  slider: document.querySelector("#step-slider"),
  stepList: document.querySelector("#step-list"),
  stageTitle: document.querySelector("#stage-title"),
  stageCaption: document.querySelector("#stage-caption"),
  candidateCount: document.querySelector("#candidate-count"),
  distanceCount: document.querySelector("#distance-count"),
  recallHint: document.querySelector("#recall-hint"),
  note: document.querySelector("#floating-note"),
  stepExplanation: document.querySelector("#step-explanation"),
  speedExplanation: document.querySelector("#speed-explanation"),
  pitfallExplanation: document.querySelector("#pitfall-explanation"),
};

const W = canvas.width;
const H = canvas.height;
const plot = { x: 62, y: 42, w: 856, h: 512 };
const query = { id: "Q", x: 0.66, y: 0.48 };
const points = [
  { id: "A", x: 0.18, y: 0.72 }, { id: "B", x: 0.24, y: 0.64 },
  { id: "C", x: 0.29, y: 0.77 }, { id: "D", x: 0.17, y: 0.39 },
  { id: "E", x: 0.31, y: 0.43 }, { id: "F", x: 0.42, y: 0.64 },
  { id: "G", x: 0.45, y: 0.29 }, { id: "H", x: 0.54, y: 0.72 },
  { id: "I", x: 0.61, y: 0.58 }, { id: "J", x: 0.68, y: 0.54 },
  { id: "K", x: 0.71, y: 0.42 }, { id: "L", x: 0.76, y: 0.34 },
  { id: "M", x: 0.82, y: 0.52 }, { id: "N", x: 0.87, y: 0.68 },
  { id: "O", x: 0.58, y: 0.23 }, { id: "P", x: 0.36, y: 0.18 },
  { id: "R", x: 0.51, y: 0.44 }, { id: "S", x: 0.73, y: 0.76 },
];

const nearestIds = ["J", "K", "I"];
const lshCandidates = ["I", "J", "K", "M"];
const treeVisited = ["R", "I", "J", "K"];
const hnswPath = ["B", "F", "H", "I", "J"];

const algorithms = {
  lsh: {
    title: "LSH 局部敏感哈希",
    description: "LSH 用多个随机超平面给向量生成签名。签名相同的点进入同一个桶，查询只扫描少量桶。",
    steps: [
      {
        title: "先把向量放到空间里",
        caption: "每个蓝点是一条数据向量，红点是查询向量。",
        note: "目标是找到离红点最近的蓝点。如果逐个计算，18 个点要算 18 次；真实系统里可能是千万级。",
        explain: "二维图只是为了看懂。实际向量可能有 384、768 或更高维，但“距离越近越相似”的想法一样。",
        speed: "此刻还没有加速，作为完整扫描的基线。",
        pitfall: "ANN 不是改变相似度定义，而是改变先看哪些候选。",
        candidates: [],
        distances: 18,
        recall: "精确",
      },
      {
        title: "画出随机投影线",
        caption: "每条线把空间切成两边，点在线的哪一侧会形成 0/1 位。",
        note: "LSH 的关键直觉：相近点被随机线切开的概率更低，因此更容易得到相同签名。",
        explain: "这里用三条线生成 3 位签名，例如 101。真实应用常用多组哈希表提高召回。",
        speed: "只计算哈希位，不需要跟所有点做精确距离比较。",
        pitfall: "投影线太少，桶会很大；投影线太多，真正近邻可能被分散。",
        candidates: [],
        distances: 0,
        recall: "建桶",
      },
      {
        title: "给每个点生成哈希签名",
        caption: "颜色代表不同桶，相同签名的点被放到一起。",
        note: "现在每个向量都变成了短签名。相似向量大概率共享签名，但不是绝对保证。",
        explain: "哈希签名是 LSH 的索引键。查询时也会用同样的随机线计算签名。",
        speed: "建好桶之后，查找候选接近 O(1)，瓶颈变成桶内距离计算。",
        pitfall: "LSH 的效果依赖距离类型，比如余弦距离常配随机超平面哈希。",
        candidates: [],
        distances: 0,
        recall: "分桶",
      },
      {
        title: "查询点也落入一个桶",
        caption: "红点的签名决定它优先查看哪个桶。",
        note: "红点附近的 J、K、I 落在同一候选区域，LSH 希望把这些点提前捞出来。",
        explain: "查询向量不再扫描全库，而是先定位到一个或几个可能相关的桶。",
        speed: "从 18 个点缩小到 4 个候选，距离计算数量明显下降。",
        pitfall: "如果最近邻刚好在桶边界另一侧，单表 LSH 可能漏掉它。",
        candidates: lshCandidates,
        distances: 4,
        recall: "较高",
      },
      {
        title: "只在候选桶内计算距离",
        caption: "绿色点是本次真正参与精排的候选。",
        note: "ANN 的“近似”发生在这里：没进候选集的点暂时不算距离。",
        explain: "最终结果仍然由精确距离排序，只是精排对象从全库变成了候选集。",
        speed: "计算量约为全扫描的 4/18。大库中这个比例可能更夸张。",
        pitfall: "召回率要用真实最近邻评估，不能只看速度。",
        candidates: lshCandidates,
        distances: 4,
        recall: "命中 3/3",
      },
      {
        title: "得到近似最近邻结果",
        caption: "高亮候选中的前三名，完成一次 LSH 查询。",
        note: "这次查询找到了真实最近的 J、K、I。换一组线或数据分布，结果可能不同。",
        explain: "工程中会调节哈希表数量、签名长度和多探测策略，在召回和速度之间取平衡。",
        speed: "速度来自“少算距离”，召回来自“相似点更可能同桶”的概率保证。",
        pitfall: "LSH 通常需要针对数据分布调参，桶过斜会导致冷热不均。",
        candidates: nearestIds,
        distances: 4,
        recall: "命中 3/3",
      },
    ],
  },
  tree: {
    title: "KD-Tree 树结构方法",
    description: "KD-Tree 轮流按 x/y 维度切分空间。低维时能剪掉很多区域，高维时剪枝能力会下降。",
    steps: [
      {
        title: "从二维点云开始",
        caption: "KD-Tree 会递归选择一个维度，把点分成左右子树。",
        note: "树方法很适合用二维图理解：每一次切分都是一条横线或竖线。",
        explain: "KD-Tree 是空间划分树。节点保存一个点，左/右子树代表某个维度上的两侧。",
        speed: "如果能剪掉远离查询点的区域，就不用访问那里的点。",
        pitfall: "这类树最怕维度很高，因为空间会变得稀疏。",
        candidates: [],
        distances: 18,
        recall: "基线",
      },
      {
        title: "第一层：按 x 维切开",
        caption: "中位点附近的竖线把空间分成左右两半。",
        note: "选中位数能让树尽量平衡，避免一边很深一边很浅。",
        explain: "第一层通常选方差较大或轮换的维度切分，这里用 x 维演示。",
        speed: "查询时先判断红点在切分线哪侧，优先走更可能近的一侧。",
        pitfall: "树平衡不等于搜索一定快，数据分布和查询位置同样重要。",
        candidates: ["R"],
        distances: 1,
        recall: "下降中",
      },
      {
        title: "第二层：按 y 维继续切",
        caption: "每个子区域再用横线切开，形成更小的矩形区域。",
        note: "轮流切维度可以把空间组织成一棵可搜索的树。",
        explain: "查询沿着树走到叶子，相当于先找到包含查询点的小区域。",
        speed: "越靠近叶子，候选范围越小。",
        pitfall: "如果数据不是低维几何结构，矩形边界不一定能代表相似性。",
        candidates: ["R", "I"],
        distances: 2,
        recall: "缩小",
      },
      {
        title: "沿查询所在分支向下走",
        caption: "绿色路径展示搜索优先访问的节点。",
        note: "红点位于右中区域，因此搜索会优先进入包含它的子树。",
        explain: "这一步类似二分搜索，但目标是空间距离，不只是大小比较。",
        speed: "先拿到一个当前最佳距离，再用它反过来判断哪些区域可剪枝。",
        pitfall: "只走单一路径可能错过边界另一侧的近邻，所以需要回溯检查。",
        candidates: ["R", "I", "J"],
        distances: 3,
        recall: "接近",
      },
      {
        title: "用当前半径剪枝",
        caption: "虚线圆是当前最佳距离，和圆不相交的区域可跳过。",
        note: "如果某个矩形区域离查询点的最短距离都超过当前最佳，就不用进去。",
        explain: "剪枝是 KD-Tree 的主要加速来源。它不是不精确，精确 KD-Tree 可以找真最近邻。",
        speed: "本例只访问 4 个关键节点，比全扫描少很多。",
        pitfall: "高维时查询球会碰到太多区域，剪枝变少，速度接近全扫描。",
        candidates: treeVisited,
        distances: 4,
        recall: "命中",
      },
      {
        title: "输出低维下的近邻",
        caption: "KD-Tree 找到离查询点最近的候选。",
        note: "在二维、三维等低维场景，树结构直观且高效；在向量检索高维场景，常被图索引或量化方法替代。",
        explain: "KD-Tree 的教育价值很强：它清楚展示了“索引 = 组织搜索空间”。",
        speed: "速度来自区域剪枝，而不是哈希概率。",
        pitfall: "把 KD-Tree 直接搬到高维文本 embedding 上，通常不会是最佳选择。",
        candidates: nearestIds,
        distances: 4,
        recall: "命中 3/3",
      },
    ],
  },
  hnsw: {
    title: "HNSW 图索引方法",
    description: "HNSW 构建分层近邻图。查询从稀疏高层快速接近目标，再在底层精细搜索。",
    steps: [
      {
        title: "先把点连成近邻图",
        caption: "边连接彼此相近的向量，搜索可以沿边移动。",
        note: "图索引不切空间，也不把点硬塞进桶，而是把“谁离谁近”直接编码成边。",
        explain: "HNSW 的底层类似一个小世界图：局部有近邻边，整体有少量长跳边。",
        speed: "搜索不需要扫全库，只要沿着越来越近的节点前进。",
        pitfall: "图构建质量会影响召回，边太少容易迷路，边太多耗内存。",
        candidates: [],
        distances: 18,
        recall: "基线",
      },
      {
        title: "高层图提供远距离跳转",
        caption: "少数节点出现在高层，形成稀疏导航图。",
        note: "高层像高速路，先粗略靠近查询点所在区域。",
        explain: "HNSW 是层级图。越高层节点越少，适合快速定位大方向。",
        speed: "一次跳转可能跨过很多普通节点。",
        pitfall: "入口点选得不好也没关系，贪心会逐步修正，但仍受图质量影响。",
        candidates: ["B"],
        distances: 1,
        recall: "导航",
      },
      {
        title: "从入口点开始贪心搜索",
        caption: "每一步比较邻居，移动到更接近查询点的节点。",
        note: "从 B 出发，发现 F 更近，就跳到 F；再继续看 F 的邻居。",
        explain: "贪心规则很简单：只要邻居更近，就移动过去。",
        speed: "距离计算集中在当前节点的邻居，而不是所有点。",
        pitfall: "纯贪心可能陷入局部最优，所以 HNSW 底层会保留候选队列。",
        candidates: ["B", "F"],
        distances: 5,
        recall: "前进",
      },
      {
        title: "逐层下降，局部变精细",
        caption: "越到底层，边越密，搜索越细。",
        note: "HNSW 会把高层找到的位置作为下一层入口，继续靠近查询点。",
        explain: "分层结构让它兼顾远距离导航和近距离精排。",
        speed: "先粗后细，避免从底层图随机乱走。",
        pitfall: "参数 M 和 efSearch 控制边数与搜索宽度，直接影响内存、速度、召回。",
        candidates: ["B", "F", "H", "I"],
        distances: 9,
        recall: "提高",
      },
      {
        title: "候选队列保留多个可能答案",
        caption: "绿色节点是访问路径，浅绿色节点是扩展过的候选邻居。",
        note: "不是只相信一条路径，而是在局部保留一批最有希望的点。",
        explain: "efSearch 越大，保留候选越多，召回越高，但查询越慢。",
        speed: "HNSW 的强项是用较少访问量换到很高召回。",
        pitfall: "追求极致召回时，efSearch 会增大，延迟也会上升。",
        candidates: ["B", "F", "H", "I", "J", "K", "M"],
        distances: 11,
        recall: "很高",
      },
      {
        title: "输出近邻，速度与召回均衡",
        caption: "最终在访问过的局部图中选出最近点。",
        note: "这就是为什么 HNSW 是很多向量数据库的默认强力选项：直觉简单，工程效果好。",
        explain: "最终排序仍用真实距离，只是 HNSW 用图搜索快速找到高质量候选。",
        speed: "速度来自小世界图的短路径和有限候选队列。",
        pitfall: "代价是索引构建较慢、占用内存较高，并且删除/更新比简单结构复杂。",
        candidates: nearestIds,
        distances: 11,
        recall: "命中 3/3",
      },
    ],
  },
};

let currentAlgo = "lsh";
let currentStep = 0;
let playTimer = null;

function toCanvas(p) {
  return {
    x: plot.x + p.x * plot.w,
    y: plot.y + (1 - p.y) * plot.h,
  };
}

function distance(a, b = query) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function pointById(id) {
  return points.find((p) => p.id === id);
}

function lineSide(line, p) {
  return line.a * p.x + line.b * p.y + line.c > 0 ? 1 : 0;
}

const lshLines = [
  { a: 1.0, b: 0.35, c: -0.72 },
  { a: -0.25, b: 1.0, c: -0.52 },
  { a: 0.85, b: -0.75, c: -0.08 },
];

function signature(p) {
  return lshLines.map((line) => lineSide(line, p)).join("");
}

const palette = {
  "000": "#9b6ad6",
  "001": "#348bd6",
  "010": "#2aa879",
  "011": "#c69224",
  "100": "#df6b52",
  "101": "#4a9b52",
  "110": "#6c7edb",
  "111": "#c45a9f",
};

function clearCanvas() {
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.strokeStyle = "#d8d0c3";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
  ctx.fillStyle = "#8a95a6";
  ctx.font = "13px Inter, sans-serif";
  ctx.fillText("向量空间", plot.x, plot.y - 16);
  ctx.restore();
}

function drawPoint(p, options = {}) {
  const pos = toCanvas(p);
  const r = options.radius || 8;
  ctx.save();
  if (options.glow) {
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r + 12, 0, Math.PI * 2);
    ctx.fillStyle = options.glow;
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
  ctx.fillStyle = options.fill || "#2f80ed";
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 3;
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = options.labelColor || "#1d2433";
  ctx.font = "700 12px Inter, sans-serif";
  ctx.fillText(p.id, pos.x + 11, pos.y - 9);
  ctx.restore();
}

function drawAllPoints(candidates = [], result = []) {
  const candidateSet = new Set(candidates);
  const resultSet = new Set(result);
  points.forEach((p) => {
    const isCandidate = candidateSet.has(p.id);
    const isResult = resultSet.has(p.id);
    drawPoint(p, {
      fill: isResult ? "#1f9d72" : isCandidate ? "#68c69d" : "#2f80ed",
      radius: isResult ? 10 : 8,
      glow: isCandidate ? "rgba(31, 157, 114, 0.13)" : null,
    });
  });
  drawPoint(query, {
    fill: "#dc4c3f",
    radius: 11,
    glow: "rgba(220, 76, 63, 0.18)",
    labelColor: "#7b1d17",
  });
}

function drawLshLine(line, color = "#d59b21") {
  const pts = [];
  for (const x of [0, 1]) {
    const y = (-line.a * x - line.c) / line.b;
    if (y >= 0 && y <= 1) pts.push({ x, y });
  }
  for (const y of [0, 1]) {
    const x = (-line.b * y - line.c) / line.a;
    if (x >= 0 && x <= 1) pts.push({ x, y });
  }
  if (pts.length < 2) return;
  const a = toCanvas(pts[0]);
  const b = toCanvas(pts[1]);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 7]);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function drawBucketLabels() {
  points.forEach((p) => {
    const pos = toCanvas(p);
    const sig = signature(p);
    ctx.save();
    ctx.fillStyle = palette[sig];
    ctx.globalAlpha = 0.16;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#1d2433";
    ctx.font = "700 11px Inter, sans-serif";
    ctx.fillText(sig, pos.x - 10, pos.y + 27);
    ctx.restore();
  });
}

function drawCandidateHull(ids, color = "rgba(31, 157, 114, 0.16)") {
  if (!ids.length) return;
  const selected = ids.map(pointById).filter(Boolean).map(toCanvas);
  const xs = selected.map((p) => p.x);
  const ys = selected.map((p) => p.y);
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = "rgba(31, 157, 114, 0.45)";
  ctx.lineWidth = 2;
  roundRect(Math.min(...xs) - 36, Math.min(...ys) - 36, Math.max(...xs) - Math.min(...xs) + 72, Math.max(...ys) - Math.min(...ys) + 72, 18);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawKdPartitions(step) {
  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#d59b21";
  if (step >= 1) {
    drawPartitionLine("v", 0.52, plot.y, plot.y + plot.h, "x 中位切分");
  }
  if (step >= 2) {
    drawPartitionLine("h", 0.58, plot.x, plot.x + plot.w * 0.52, "左子树 y 切分");
    drawPartitionLine("h", 0.46, plot.x + plot.w * 0.52, plot.x + plot.w, "右子树 y 切分");
  }
  if (step >= 3) {
    ctx.strokeStyle = "#7255d9";
    drawPartitionLine("v", 0.70, plot.y + plot.h * 0.54, plot.y + plot.h, "局部 x 切分");
  }
  ctx.restore();
}

function drawPartitionLine(kind, value, from, to, label) {
  ctx.save();
  ctx.setLineDash([7, 7]);
  ctx.beginPath();
  if (kind === "v") {
    const x = plot.x + value * plot.w;
    ctx.moveTo(x, from);
    ctx.lineTo(x, to);
    ctx.stroke();
    ctx.fillStyle = "#6c6558";
    ctx.font = "700 12px Inter, sans-serif";
    ctx.fillText(label, x + 8, Math.max(from + 18, plot.y + 18));
  } else {
    const y = plot.y + (1 - value) * plot.h;
    ctx.moveTo(from, y);
    ctx.lineTo(to, y);
    ctx.stroke();
    ctx.fillStyle = "#6c6558";
    ctx.font = "700 12px Inter, sans-serif";
    ctx.fillText(label, from + 8, y - 8);
  }
  ctx.restore();
}

function drawRadiusToNearest() {
  const q = toCanvas(query);
  const near = toCanvas(pointById("J"));
  const r = Math.hypot(q.x - near.x, q.y - near.y);
  ctx.save();
  ctx.strokeStyle = "rgba(220, 76, 63, 0.55)";
  ctx.setLineDash([5, 7]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(q.x, q.y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

const graphEdges = [
  ["A", "B"], ["A", "D"], ["B", "C"], ["B", "F"], ["C", "F"],
  ["D", "E"], ["E", "G"], ["E", "R"], ["F", "H"], ["F", "R"],
  ["G", "O"], ["G", "P"], ["H", "I"], ["H", "S"], ["I", "J"],
  ["I", "R"], ["J", "K"], ["J", "M"], ["K", "L"], ["K", "M"],
  ["M", "N"], ["N", "S"], ["O", "L"], ["P", "G"], ["R", "K"],
];
const upperEdges = [["B", "H"], ["H", "M"], ["F", "J"], ["B", "N"]];

function drawGraph(step) {
  const edges = step >= 1 ? graphEdges.concat(upperEdges) : graphEdges;
  ctx.save();
  edges.forEach(([aId, bId]) => {
    const a = toCanvas(pointById(aId));
    const b = toCanvas(pointById(bId));
    const upper = upperEdges.some(([u, v]) => (u === aId && v === bId) || (u === bId && v === aId));
    ctx.strokeStyle = upper ? "rgba(114, 85, 217, 0.48)" : "rgba(99, 112, 131, 0.24)";
    ctx.lineWidth = upper ? 3 : 1.6;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });
  ctx.restore();
}

function drawPath(ids) {
  if (ids.length < 2) return;
  ctx.save();
  ctx.strokeStyle = "#1f9d72";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ids.forEach((id, index) => {
    const p = toCanvas(pointById(id));
    if (index === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  ctx.restore();
}

function renderLsh(stepIndex, data) {
  clearCanvas();
  const showLines = stepIndex >= 1;
  const showBuckets = stepIndex >= 2;
  const showHull = stepIndex >= 3;
  const result = stepIndex >= 5 ? nearestIds : [];
  if (showLines) lshLines.forEach((line) => drawLshLine(line));
  if (showBuckets) drawBucketLabels();
  if (showHull) drawCandidateHull(lshCandidates);
  drawAllPoints(data.candidates, result);
  if (stepIndex >= 3) drawDashedDistanceLines(lshCandidates);
}

function renderTree(stepIndex, data) {
  clearCanvas();
  drawKdPartitions(stepIndex);
  if (stepIndex >= 4) drawRadiusToNearest();
  const result = stepIndex >= 5 ? nearestIds : [];
  drawAllPoints(data.candidates, result);
  if (stepIndex >= 3) drawSearchOrder(data.candidates);
}

function renderHnsw(stepIndex, data) {
  clearCanvas();
  drawGraph(stepIndex);
  const path = hnswPath.slice(0, Math.min(stepIndex + 1, hnswPath.length));
  if (stepIndex >= 2) drawPath(path);
  const result = stepIndex >= 5 ? nearestIds : [];
  drawAllPoints(data.candidates, result);
  if (stepIndex >= 2) drawDashedDistanceLines(data.candidates.slice(-3));
}

function drawDashedDistanceLines(ids) {
  const q = toCanvas(query);
  ctx.save();
  ctx.strokeStyle = "rgba(220, 76, 63, 0.34)";
  ctx.lineWidth = 1.6;
  ctx.setLineDash([5, 5]);
  ids.forEach((id) => {
    const p = toCanvas(pointById(id));
    ctx.beginPath();
    ctx.moveTo(q.x, q.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  });
  ctx.restore();
}

function drawSearchOrder(ids) {
  ctx.save();
  ids.forEach((id, index) => {
    const p = toCanvas(pointById(id));
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#1f9d72";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(p.x - 16, p.y + 15, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#1f9d72";
    ctx.font = "800 11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(index + 1), p.x - 16, p.y + 15);
  });
  ctx.restore();
}

function render() {
  const algo = algorithms[currentAlgo];
  const step = algo.steps[currentStep];
  els.title.textContent = algo.title;
  els.description.textContent = algo.description;
  els.stepCounter.textContent = `${currentStep + 1} / ${algo.steps.length}`;
  els.slider.max = String(algo.steps.length - 1);
  els.slider.value = String(currentStep);
  els.stageTitle.textContent = step.title;
  els.stageCaption.textContent = step.caption;
  els.candidateCount.textContent = String(step.candidates.length);
  els.distanceCount.textContent = String(step.distances);
  els.recallHint.textContent = step.recall;
  els.note.textContent = step.note;
  els.stepExplanation.textContent = step.explain;
  els.speedExplanation.textContent = step.speed;
  els.pitfallExplanation.textContent = step.pitfall;

  els.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.algo === currentAlgo));
  renderStepList(algo.steps);

  if (currentAlgo === "lsh") renderLsh(currentStep, step);
  if (currentAlgo === "tree") renderTree(currentStep, step);
  if (currentAlgo === "hnsw") renderHnsw(currentStep, step);
}

function renderStepList(steps) {
  els.stepList.innerHTML = "";
  steps.forEach((step, index) => {
    const li = document.createElement("li");
    li.className = index === currentStep ? "active" : "";
    li.innerHTML = `<span>${index + 1}</span><div>${step.title}</div>`;
    li.addEventListener("click", () => {
      currentStep = index;
      stopPlaying();
      render();
    });
    els.stepList.appendChild(li);
  });
}

function nextStep() {
  const steps = algorithms[currentAlgo].steps;
  currentStep = (currentStep + 1) % steps.length;
  render();
}

function prevStep() {
  const steps = algorithms[currentAlgo].steps;
  currentStep = (currentStep - 1 + steps.length) % steps.length;
  render();
}

function stopPlaying() {
  if (!playTimer) return;
  clearInterval(playTimer);
  playTimer = null;
  els.play.textContent = "播放";
}

function togglePlaying() {
  if (playTimer) {
    stopPlaying();
    return;
  }
  els.play.textContent = "暂停";
  playTimer = setInterval(nextStep, 1800);
}

els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    currentAlgo = tab.dataset.algo;
    currentStep = 0;
    stopPlaying();
    render();
  });
});

els.prev.addEventListener("click", () => {
  prevStep();
  stopPlaying();
});
els.next.addEventListener("click", () => {
  nextStep();
  stopPlaying();
});
els.play.addEventListener("click", togglePlaying);
els.slider.addEventListener("input", (event) => {
  currentStep = Number(event.target.value);
  stopPlaying();
  render();
});

render();
