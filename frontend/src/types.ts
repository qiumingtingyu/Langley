export type ActiveView = "chat" | "memory" | "knowledge";

export type RunStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";

export interface Conversation {
  id: number;
  title: string | null;
  created_at: string;
  updated_at: string;
  last_message_at: string | null;
}

export interface Message {
  id: number;
  sequence_no: number;
  role: "USER" | "ASSISTANT";
  content: string;
  run_id: number | null;
  regenerated_from_message_id: number | null;
  created_at: string;
}

export interface Run {
  id: number;
  input_message_id: number;
  attempt_no: number;
  status: RunStatus;
  started_at: string | null;
  finished_at: string | null;
  error_code: string | null;
}

export interface StreamState {
  conversationId: number;
  runId: number;
  viewRevision: number;
  content: string;
}
