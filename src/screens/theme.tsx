// theme.tsx
// Central design tokens for the eKYC app. Import these everywhere instead
// of hardcoding colors/spacing so the whole app stays visually consistent.

// Palette matches the indigo direction already built in App.tsx
// (Cpu/ShieldCheck header, gradient scan button) so all screens read as
// one consistent product instead of two competing designs.
export const colors = {
  bg: '#090D16',          // near-black — main background (matches App.tsx)
  bgElevated: '#111827',  // cards, sheets, modals
  bgGlass: 'rgba(255,255,255,0.05)', // glassmorphism panels over camera

  primary: '#6366F1',     // indigo — primary actions, active states
  primaryAlt: '#4F46E5',  // gradient end color for buttons
  primaryDim: '#6366F133',

  danger: '#EF4444',      // rejection / spoof detected / errors only
  dangerDim: '#EF444422',

  success: '#10B981',
  successDim: '#10B98122',

  warning: '#F59E0B',

  textPrimary: '#F8FAFC',
  textSecondary: '#94A3B8',
  textMuted: '#64748B',

  border: '#1E293B',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const radius = {
  sm: 8,
  md: 16,
  lg: 24,
  pill: 999,
};

export const typography = {
  h1: { fontSize: 30, fontWeight: '700' as const, letterSpacing: -0.5 },
  h2: { fontSize: 22, fontWeight: '700' as const, letterSpacing: -0.3 },
  body: { fontSize: 15, fontWeight: '400' as const },
  bodyBold: { fontSize: 15, fontWeight: '600' as const },
  caption: { fontSize: 13, fontWeight: '400' as const },
  label: { fontSize: 12, fontWeight: '600' as const, letterSpacing: 0.6 },
};