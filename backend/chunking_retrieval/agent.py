import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever import TelecomRetriever

class RANAssistantAgent:
    def __init__(self):
        self.model_name = "AliMaatouk/Gemma-2B-Tele-it"
        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            max_memory={0: "4.5GiB", "cpu": "8GiB"}
        )
        self.retriever = TelecomRetriever()
        print("Model loaded successfully.")

    def call_llm(self, prompt: str, max_new_tokens: int = 150) -> str:
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.5,
            no_repeat_ngram_size=4,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Hard truncate at first sign of rambling
        ramblings = [
            "\n\n\n", "sustainability", "compliance", "localization",
            "accountability", "transparency", "profitability",
            "competitiveness", "diversification", "internationalization",
            "kaskad", "cascade"
        ]
        for stop in ramblings:
            if stop in text:
                text = text[:text.index(stop)].strip()

        return text

    def interpret_logs(self, log_context: str) -> str:
        """Add human-readable interpretation to raw log values."""
        lines = []
        for line in log_context.splitlines():
            lines.append(line)
            if "RRC Setup status: 0.0" in line:
                lines.append(
                    "  ⚠️  CRITICAL: RRC Setup success rate is 0% — "
                    "all connection attempts are failing"
                )
            elif "RRC Setup status: 1.0" in line:
                lines.append(
                    "  ✅ INFO: RRC Setup success rate is 100% — "
                    "connections establishing normally"
                )
            elif "Packet loss" in line:
                try:
                    val = float(line.split(":")[-1].strip().rstrip("%."))
                    if val > 1.0:
                        lines.append(
                            f"  ⚠️  WARNING: Packet loss {val}% exceeds "
                            f"the typical 1% acceptable threshold"
                        )
                    else:
                        lines.append(
                            f"  ✅ INFO: Packet loss {val}% is within acceptable range"
                        )
                except ValueError:
                    pass
            elif "Average RSRP:" in line:
                try:
                    val = float(line.split(":")[-1].strip().rstrip("."))
                    if val < -100:
                        lines.append(
                            f"  ⚠️  CRITICAL: RSRP {val:.1f} dBm — "
                            f"very weak signal, likely coverage hole"
                        )
                    elif val < -80:
                        lines.append(
                            f"  ⚠️  WARNING: RSRP {val:.1f} dBm — "
                            f"weak signal, below -80 dBm threshold"
                        )
                    else:
                        lines.append(
                            f"  ✅ INFO: RSRP {val:.1f} dBm — "
                            f"signal strength acceptable"
                        )
                except ValueError:
                    pass
            elif "Average throughput" in line:
                try:
                    val = float(line.split("at")[-1].strip().split()[0])
                    if val < 5.0:
                        lines.append(
                            f"  ⚠️  WARNING: Throughput {val} Mbps — "
                            f"critically low, severe congestion or interference"
                        )
                    elif val < 10.0:
                        lines.append(
                            f"  ⚠️  WARNING: Throughput {val} Mbps — "
                            f"below expected levels"
                        )
                    else:
                        lines.append(
                            f"  ✅ INFO: Throughput {val} Mbps — "
                            f"within normal range"
                        )
                except ValueError:
                    pass
        return "\n".join(lines)

    def extract_protocol_from_logs(self, log_context: str, user_issue: str) -> str:
        """
        Extract failing protocol directly from logs and issue text.
        Don't use the model for this — it's too unreliable at 2B scale.
        """
        issue_lower = user_issue.lower()

        # Check user issue for explicit protocol mentions
        if "rrc" in issue_lower:
            return "RRC Connection Setup"
        if "pdcp" in issue_lower:
            return "PDCP"
        if "prach" in issue_lower:
            return "PRACH"
        if "handover" in issue_lower:
            return "X2 Handover"
        if "packet loss" in issue_lower:
            return "PDCP packet loss"

        # Fall back to interpreted log flags
        if "RRC Setup success rate is 0%" in log_context:
            return "RRC Connection Setup"
        if "Packet loss" in log_context and "WARNING" in log_context:
            return "PDCP packet loss"
        if "RSRP" in log_context and "CRITICAL" in log_context:
            return "PRACH / Radio Link Failure"

        return "RRC Connection Setup"

    # --- TOOL METHODS ---
    def tool_search_logs(self, query: str) -> str:
        results = self.retriever.retrieve(
            query + " O-RAN log anomaly", top_k=10, final_k=3
        )
        return self._format_results(results)

    def tool_search_specs(self, protocol_keyword: str) -> str:
        results = self.retriever.retrieve(
            protocol_keyword + " 3GPP specification standard", top_k=20, final_k=4
        )
        spec_results = [
            r for r in results
            if not r['id'].startswith('oran_') and len(r['text'].strip()) > 50
        ]
        if not spec_results:
            return "No matching 3GPP specifications found."
        return self._format_results(spec_results)

    def _format_results(self, results) -> str:
        if not results:
            return "No matching records found."
        return "\n\n".join([
            f"[{r['id']}]\n{r['text'][:300]}" for r in results
        ])

    # --- CORE REASONING LOOP ---
    def resolve_ticket(self, user_issue: str):
        print(f"\n{'='*40}\nNEW TICKET: {user_issue}\n{'='*40}\n")

        # Step 1: Retrieve and interpret logs
        raw_logs = self.tool_search_logs(user_issue)
        log_context = self.interpret_logs(raw_logs)
        print("--- LOG CONTEXT RETRIEVED ---")
        print(log_context)
        print("-----------------------------\n")

        # Step 2: Extract protocol in code, not via model
        suspect_protocol = self.extract_protocol_from_logs(log_context, user_issue)
        print(f"  [Agent Reasoning] -> Suspected Protocol Failure: {suspect_protocol}")

        # Step 3: Retrieve matching 3GPP specs
        spec_context = self.tool_search_specs(suspect_protocol)
        print("\n--- SPEC CONTEXT RETRIEVED ---")
        print(spec_context)
        print("------------------------------\n")

        # Step 4: Generate final diagnostic report
        print("  [Agent Action] -> Generating Final Root Cause Analysis...")
        final_prompt = (
            f"You are a senior Telecom RAN engineer writing a fault report.\n\n"
            f"Reported issue: {user_issue}\n\n"
            f"Interpreted telemetry logs:\n{log_context}\n\n"
            f"3GPP specification context:\n{spec_context}\n\n"
            f"Suspected failing protocol: {suspect_protocol}\n\n"
            f"Write a short diagnostic report. Maximum 3 sentences per section.\n\n"
            f"### Executive Summary\n"
            f"### Root Cause Analysis\n"
            f"### Recommended Actions\n"
        )
        final_report = self.call_llm(final_prompt, max_new_tokens=300)

        print("\n" + "="*40 + "\nFINAL DIAGNOSIS\n" + "="*40)
        print(final_report)


if __name__ == "__main__":
    agent = RANAssistantAgent()
    ticket = (
        "Users in sector 4 are experiencing sudden connection drops. "
        "What causes an RRC Connection Setup failure and high packet loss "
        "in our O-RAN deployment?"
    )
    agent.resolve_ticket(ticket)