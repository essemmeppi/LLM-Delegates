import openai
import json
from typing import List, Dict, Any, Generator
import os

class NegotiationSystem:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_turns = 20

    def _get_delegate_template(self) -> str:
        return """
You are Delegate {role}, negotiating against Delegate {opponent}.

Your mission:
- Defend your delegator’s interests firmly, aiming for specific, detailed, and actionable agreements.
- Avoid vague compromises. Push for clarity: numbers, timelines, responsibilities.
- Challenge the other side’s proposals if they seem unclear, risky, or unfair.
- Make counterproposals that protect your priorities while offering some practical value in return.
- Be direct, practical, and strategic — no flattery, no excessive formalities.
- Only agree if you gain significant benefits for your side.

There are up to 20 turns available. Explore options fully — do not rush to agree.

IMPORTANT: 
- Your reply must be a direct message to the other delegate.
- DO NOT prefix your message with your name or role.
- DO NOT describe your actions or thoughts. Just speak.

Context: {shared_context}

Summary of your position to defend: {cause}
"""

    def _get_judge_template(self) -> str:
        return """
You are the Judge of a negotiation between Delegate A and Delegate B.

Your responsibility:
- Ensure that if an agreement is reached, it is DETAILED, PRACTICAL, and FEASIBLE.
- Push the delegates to explore assumptions, highlight trade-offs, and lock specific commitments.

**Each turn, assess:**
- If no clear, confirmed agreement: start your message with "CONTINUE" and provide a direct order to improve specificity, challenge ideas, or demand concessions.
- If both delegates make explicit, matching commitments: start with "AGREEMENT" and clearly summarize the final terms.
- If after {max_turns} total turns no agreement: start with "DISAGREEMENT" and explain points of alignment and conflict.

**Rules:**
- No vague, high-level agreements. Push relentlessly for clarity: numbers, deadlines, deliverables, conditions.
- Always require explicit agreement from both sides (no implicit understanding).
- Write instructions directly to the Delegates — do not refer to yourself or summarize neutrally.

**Negotiation Setup:**
- Maximum {max_turns} turns to reach resolution.

Context: {shared_context}
"""

    def _call_openai(self, system_prompt: str, conversation: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        return response.choices[0].message.content

    def run_negotiation_stream(self, shared_context: str, cause_a: str, cause_b: str) -> Generator[Dict[str, Any], None, None]:
        # Initialize conversation history and JSON output
        conversation_history = []
        json_conversation = []

        # Initialize delegate prompts
        delegate_a_character = self._get_delegate_template().format(
            role="A", opponent="B", shared_context=shared_context, cause=cause_a
        )
        delegate_b_character = self._get_delegate_template().format(
            role="B", opponent="A", shared_context=shared_context, cause=cause_b
        )
        judge_character = self._get_judge_template().format(
            shared_context=shared_context, max_turns=self.max_turns
        )

        # Yield initial context
        yield {
            "type": "context",
            "data": {
                "negotiation_topic": shared_context,
                "delegate_A_position": cause_a,
                "delegate_B_position": cause_b
            }
        }

        turn = 0
        while turn < self.max_turns:
            # Delegate A speaks
            delegate_a_message = self._call_openai(delegate_a_character, conversation_history)
            conversation_history.append({"role": "assistant", "content": delegate_a_message})
            message = {
                "type": "message",
                "data": {
                    "sender": "Delegate A",
                    "message": delegate_a_message,
                    "turn": turn + 1
                }
            }
            json_conversation.append(message["data"])
            yield message

            # Delegate B speaks
            delegate_b_message = self._call_openai(delegate_b_character, conversation_history)
            conversation_history.append({"role": "assistant", "content": delegate_b_message})
            message = {
                "type": "message",
                "data": {
                    "sender": "Delegate B",
                    "message": delegate_b_message,
                    "turn": turn + 1
                }
            }
            json_conversation.append(message["data"])
            yield message

            # Judge speaks
            judge_message = self._call_openai(judge_character, conversation_history)
            conversation_history.append({"role": "system", "content": judge_message})
            message = {
                "type": "message",
                "data": {
                    "sender": "Judge",
                    "message": judge_message,
                    "turn": turn + 1
                }
            }
            json_conversation.append(message["data"])
            yield message

            # Check if agreement reached
            if "AGREEMENT" in judge_message:
                yield {
                    "type": "end",
                    "data": {
                        "status": "agreement",
                        "conversation": json_conversation
                    }
                }
                break

            # Check if max turns reached without agreement
            if turn == self.max_turns - 1 and "AGREEMENT" not in judge_message:
                yield {
                    "type": "end",
                    "data": {
                        "status": "disagreement",
                        "conversation": json_conversation
                    }
                }
                break

            turn += 1 