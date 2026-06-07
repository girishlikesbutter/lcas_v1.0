// Minimal inline icon set (stroke-based, currentColor) — avoids an icon dependency
// and keeps the bundle lean. 1.6px stroke reads well at 16–18px in the dark theme.
import type { SVGProps } from "react";

const S = (p: SVGProps<SVGSVGElement>) => ({
  width: 16, height: 16, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
  strokeWidth: 1.7, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, ...p,
});

export const IconOverview = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M3 13h8V3H3zM13 21h8V3h-8zM3 21h8v-6H3z" /></svg>
);
export const IconTree = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><circle cx="12" cy="5" r="2" /><circle cx="6" cy="19" r="2" /><circle cx="18" cy="19" r="2" /><path d="M12 7v4M12 11H6v6M12 11h6v6" /></svg>
);
export const IconFrontier = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3.5" /><path d="M12 1v3M12 20v3M1 12h3M20 12h3" /></svg>
);
export const IconRuns = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3" /><path d="M7.5 15h9" /></svg>
);
export const IconClaims = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z" /><path d="M9 12l2 2 4-4" /></svg>
);
export const IconPipelines = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><rect x="3" y="4" width="6" height="5" rx="1" /><rect x="15" y="15" width="6" height="5" rx="1" /><path d="M9 6.5h4a2 2 0 0 1 2 2v9" /></svg>
);
export const IconSubstrate = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" /><path d="M3 8v5M21 8v5" /></svg>
);
export const IconGlossary = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z" /><path d="M9 7h6M9 11h4" /></svg>
);
export const IconMachinery = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><rect x="2" y="9" width="5" height="6" rx="1" /><rect x="17" y="4" width="5" height="6" rx="1" /><rect x="17" y="14" width="5" height="6" rx="1" /><circle cx="12" cy="12" r="2.4" /><path d="M7 12h2.6M14.4 11l2.6-3M14.4 13l2.6 4" /></svg>
);
export const IconStream = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><rect x="3" y="4" width="18" height="14" rx="2" /><path d="M3 14l4-4 3 3 4-5 4 5" /><circle cx="8.5" cy="8.5" r="1.2" /></svg>
);
export const IconTerminal = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M7 9l3 3-3 3M13 15h4" /></svg>
);
export const IconChevron = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M9 6l6 6-6 6" /></svg>
);
export const IconExternal = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M14 4h6v6M20 4l-9 9M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5" /></svg>
);
export const IconPlus = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M12 5v14M5 12h14" /></svg>
);
export const IconClose = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M6 6l12 12M18 6L6 18" /></svg>
);
export const IconSearch = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></svg>
);
export const IconBlast = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M12 2l2.5 6L21 9l-5 4 1.5 7L12 16l-5.5 4L8 13 3 9l6.5-1z" /></svg>
);
export const IconWarn = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><path d="M12 9v4M12 17h.01" /></svg>
);
export const IconArrowRight = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M5 12h14M13 6l6 6-6 6" /></svg>
);
export const IconRefresh = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5" /></svg>
);
export const IconBolt = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M13 2 4 14h7l-1 8 9-12h-7z" /></svg>
);
export const IconDoc = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></svg>
);
export const IconLink = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" /><path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" /></svg>
);
// layout toggles: tiled (quadrant grid) vs tabbed (single pane + tab strip)
export const IconTiled = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><rect x="3" y="3" width="8" height="8" rx="1" /><rect x="13" y="3" width="8" height="8" rx="1" /><rect x="3" y="13" width="8" height="8" rx="1" /><rect x="13" y="13" width="8" height="8" rx="1" /></svg>
);
export const IconTabbed = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><rect x="3" y="7" width="18" height="14" rx="2" /><path d="M3 7l3-3h5l2 3" /></svg>
);
// control plane / intent queue — an inbox with a downward intake arrow
export const IconQueue = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M4 13h4l2 3h4l2-3h4" /><path d="M5 13l1.5-7h11L20 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" /><path d="M12 4v5M9.5 7.5 12 10l2.5-2.5" /></svg>
);
export const IconCheck = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M5 12l4.5 4.5L19 7" /></svg>
);
// the materials shelf — two labelled jars resting on a shelf line
export const IconMaterials = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><path d="M3 20h18" /><rect x="4.5" y="9" width="6" height="9" rx="1.2" /><rect x="13.5" y="6" width="6" height="12" rx="1.2" /><path d="M5.5 7V9M9.5 7V9M14.5 4v2M18.5 4v2M4.5 12.5h6M13.5 10h6" /></svg>
);
// copy-to-clipboard — overlapping sheets
export const IconCopy = (p: SVGProps<SVGSVGElement>) => (
  <svg {...S(p)}><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></svg>
);
