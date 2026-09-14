import BatchAuditView from "./BatchAuditView";

/**
 * V10 batch view. Paste a whole lease, get a Red/Yellow/Green breakdown of every
 * clause with drill-downs. Route: /taikyo/batch
 */
export default function TaikyoBatchPage() {
  return <BatchAuditView />;
}
