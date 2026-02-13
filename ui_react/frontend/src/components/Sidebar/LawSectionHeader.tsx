/**
 * Section header component for law list.
 *
 * Single Responsibility: Render a section divider with label and count.
 */

interface LawSectionHeaderProps {
  label: string;
  count: number;
}

export function LawSectionHeader({ label, count }: LawSectionHeaderProps) {
  return (
    <div
      className="
        flex items-center gap-1
        pt-2 pb-1 px-1
        text-xs font-medium uppercase tracking-wider
        text-apple-gray-400
      "
    >
      <span>{label}</span>
      <span>({count})</span>
    </div>
  );
}
