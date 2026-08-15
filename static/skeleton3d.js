(() => {
  "use strict";

  const PHASE_LABELS = {
    catch: "캐치",
    dip: "딥",
    rise: "라이즈",
    release: "릴리즈",
    follow_through: "팔로우스루",
  };

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const lerp = (a, b, amount) => a + ((b - a) * amount);

  class SkeletonViewer {
    constructor(options) {
      this.canvas = options.canvas;
      this.ctx = this.canvas.getContext("2d");
      this.playButton = options.playButton;
      this.slider = options.slider;
      this.timeLabel = options.timeLabel;
      this.phaseLabel = options.phaseLabel;
      this.frames = [];
      this.bones = [];
      this.hand = "right";
      this.duration = 0;
      this.time = 0;
      this.playing = false;
      this.lastTick = 0;
      this.yaw = -0.55;
      this.pitch = -0.12;
      this.zoom = 1;
      this.drag = null;

      this.playButton.addEventListener("click", () => this.toggle());
      this.slider.addEventListener("input", () => {
        this.pause();
        this.time = this.duration * (Number(this.slider.value) / 1000);
        this.render();
      });
      this.canvas.addEventListener("pointerdown", (event) => {
        this.drag = { x: event.clientX, y: event.clientY };
        this.canvas.setPointerCapture(event.pointerId);
        this.canvas.classList.add("is-dragging");
      });
      this.canvas.addEventListener("pointermove", (event) => {
        if (!this.drag) return;
        this.yaw += (event.clientX - this.drag.x) * 0.012;
        this.pitch = clamp(this.pitch + ((event.clientY - this.drag.y) * 0.009), -1.15, 0.75);
        this.drag = { x: event.clientX, y: event.clientY };
        this.render();
      });
      const endDrag = () => {
        this.drag = null;
        this.canvas.classList.remove("is-dragging");
      };
      this.canvas.addEventListener("pointerup", endDrag);
      this.canvas.addEventListener("pointercancel", endDrag);
      this.canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        this.zoom = clamp(this.zoom * (event.deltaY > 0 ? 0.92 : 1.08), 0.65, 1.8);
        this.render();
      }, { passive: false });

      if (window.ResizeObserver) {
        this.resizeObserver = new ResizeObserver(() => this.render());
        this.resizeObserver.observe(this.canvas);
      } else {
        window.addEventListener("resize", () => this.render());
      }
      this.render();
    }

    setProfile(profile) {
      this.pause();
      this.frames = Array.isArray(profile?.frames) ? profile.frames : [];
      this.bones = Array.isArray(profile?.bones) ? profile.bones : [];
      this.hand = profile?.hand === "left" ? "left" : "right";
      this.duration = Math.max(0, Number(profile?.duration ?? this.frames.at(-1)?.t ?? 0));
      this.time = 0;
      this.slider.value = "0";
      this.slider.disabled = this.frames.length <= 1;
      this.playButton.disabled = this.frames.length <= 1;
      this.render();
    }

    toggle() {
      if (this.frames.length <= 1) return;
      if (this.playing) {
        this.pause();
      } else {
        if (this.time >= this.duration) this.time = 0;
        this.playing = true;
        this.playButton.textContent = "일시정지";
        this.lastTick = performance.now();
        requestAnimationFrame((now) => this.tick(now));
      }
    }

    pause() {
      this.playing = false;
      this.playButton.textContent = "재생";
    }

    tick(now) {
      if (!this.playing) return;
      this.time += (now - this.lastTick) / 1000;
      this.lastTick = now;
      if (this.time > this.duration) this.time = 0;
      this.render();
      requestAnimationFrame((next) => this.tick(next));
    }

    frameAt(time) {
      if (!this.frames.length) return null;
      if (this.frames.length === 1 || time <= Number(this.frames[0].t)) return this.frames[0];
      let rightIndex = this.frames.findIndex((frame) => Number(frame.t) >= time);
      if (rightIndex < 0) return this.frames[this.frames.length - 1];
      if (rightIndex === 0) return this.frames[0];
      const left = this.frames[rightIndex - 1];
      const right = this.frames[rightIndex];
      const span = Math.max(0.0001, Number(right.t) - Number(left.t));
      const amount = clamp((time - Number(left.t)) / span, 0, 1);
      const landmarks = {};
      Object.keys(left.landmarks || {}).forEach((name) => {
        const a = left.landmarks[name];
        const b = right.landmarks?.[name] || a;
        landmarks[name] = [
          lerp(Number(a[0]), Number(b[0]), amount),
          lerp(Number(a[1]), Number(b[1]), amount),
          lerp(Number(a[2]), Number(b[2]), amount),
        ];
      });
      return {
        t: time,
        phase: amount < 0.5 ? left.phase : right.phase,
        landmarks,
      };
    }

    resizeCanvas() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(320, Math.round(rect.width || 640));
      const height = Math.max(320, Math.round(rect.height || 430));
      if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
        this.canvas.width = Math.round(width * dpr);
        this.canvas.height = Math.round(height * dpr);
      }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { width, height };
    }

    project(point, width, height) {
      const x = Number(point[0]);
      const y = Number(point[1]) - 0.88;
      const z = Number(point[2]);
      const cy = Math.cos(this.yaw);
      const sy = Math.sin(this.yaw);
      const cp = Math.cos(this.pitch);
      const sp = Math.sin(this.pitch);
      const rx = (x * cy) + (z * sy);
      const rz = (-x * sy) + (z * cy);
      const ry = (y * cp) - (rz * sp);
      const depth = (y * sp) + (rz * cp);
      const perspective = 3.3 / Math.max(1.8, 3.3 + depth);
      const scale = Math.min(width, height) * 0.46 * this.zoom;
      return {
        x: (width / 2) + (rx * scale * perspective),
        y: (height * 0.54) - (ry * scale * perspective),
        depth,
        perspective,
      };
    }

    drawGrid(width, height) {
      const ctx = this.ctx;
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(148, 163, 184, 0.28)";
      for (let i = -5; i <= 5; i += 1) {
        const offset = i * 0.25;
        const a = this.project([offset, 0, -1.25], width, height);
        const b = this.project([offset, 0, 1.25], width, height);
        const c = this.project([-1.25, 0, offset], width, height);
        const d = this.project([1.25, 0, offset], width, height);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke();
      }
    }

    render() {
      const { width, height } = this.resizeCanvas();
      const ctx = this.ctx;
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, "#eff6ff");
      gradient.addColorStop(0.62, "#f8fafc");
      gradient.addColorStop(1, "#e2e8f0");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      this.drawGrid(width, height);

      const frame = this.frameAt(this.time);
      if (!frame) {
        ctx.fillStyle = "#64748b";
        ctx.font = "600 15px Barlow, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("스켈레톤 데이터가 없습니다.", width / 2, height / 2);
        return;
      }

      const projected = {};
      Object.entries(frame.landmarks || {}).forEach(([name, point]) => {
        projected[name] = this.project(point, width, height);
      });
      const shootingSuffix = this.hand === "left" ? "_l" : "_r";
      const sortedBones = [...this.bones].sort((first, second) => {
        const firstDepth = ((projected[first[0]]?.depth || 0) + (projected[first[1]]?.depth || 0)) / 2;
        const secondDepth = ((projected[second[0]]?.depth || 0) + (projected[second[1]]?.depth || 0)) / 2;
        return secondDepth - firstDepth;
      });
      sortedBones.forEach(([startName, endName]) => {
        const start = projected[startName];
        const end = projected[endName];
        if (!start || !end) return;
        const shootingBone = startName.endsWith(shootingSuffix) || endName.endsWith(shootingSuffix);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.lineCap = "round";
        ctx.lineWidth = shootingBone ? 7 : 5;
        ctx.strokeStyle = shootingBone ? "#f97316" : "#1e3a5f";
        ctx.stroke();
      });
      Object.entries(projected)
        .sort((a, b) => b[1].depth - a[1].depth)
        .forEach(([name, point]) => {
          const shootingJoint = name.endsWith(shootingSuffix);
          ctx.beginPath();
          ctx.arc(point.x, point.y, name === "head" ? 10 : (shootingJoint ? 6 : 5), 0, Math.PI * 2);
          ctx.fillStyle = shootingJoint ? "#ea580c" : "#ffffff";
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = shootingJoint ? "#9a3412" : "#1e3a5f";
          ctx.stroke();
        });

      const normalized = this.duration > 0 ? this.time / this.duration : 0;
      this.slider.value = String(Math.round(normalized * 1000));
      this.timeLabel.textContent = `${this.time.toFixed(2)} / ${this.duration.toFixed(2)}s`;
      this.phaseLabel.textContent = PHASE_LABELS[frame.phase] || frame.phase || "—";
    }
  }

  window.Skeleton3D = { SkeletonViewer };
})();
