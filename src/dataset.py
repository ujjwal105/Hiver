"""
dataset.py — Build the email/reply dataset.

The dataset is *hand-authored* by the author (with AI assistance for phrasing).
It models a B2B SaaS shared-inbox support queue — the exact setting Hiver
operates in (customer email run by a team inside Gmail).

Why hand-authored + synthetic instead of a public corpus:
  - Public support corpora (e.g. Ubuntu, MultiWOZ, customer-service tweets) are
    either chat-shaped, single-domain, or lack a clean "incoming -> the reply we
    actually sent" pairing. We specifically need (email -> the *ground-truth*
    reply a human agent sent) because our evaluation is reference-aware.
  - Hand-authoring lets us control category coverage, tone variety, and the
    presence of concrete facts (order numbers, dates, plan names) that a good
    reply must correctly carry over — which is exactly what we want to test.

Honesty note (see README): this is a small, curated set. It is representative of
*shape and difficulty*, not of real-world traffic distribution. The pipeline is
built so a real exported inbox (same JSONL schema) can be dropped in unchanged.

Run:  python -m src.dataset            # writes data/emails.jsonl
      python -m src.dataset --augment  # + parametrised augmentation for volume
"""
from __future__ import annotations
import argparse, json, os, hashlib

# Each record: category, subject, incoming, reference (the reply actually sent).
# split is assigned deterministically below so results are reproducible.
CORE = [
    # ---------------- Billing ----------------
    {"category": "billing", "subject": "Charged twice this month",
     "incoming": "Hi, I just checked my card statement and I've been charged $49 twice for my Growth plan this month. My account email is dana@brightloop.io. Can you refund the duplicate charge?",
     "reference": "Hi Dana,\n\nThanks for flagging this, and sorry for the scare. I looked at the account under dana@brightloop.io and you're right — the Growth plan ($49) was billed twice on the 3rd due to a retry after a temporary gateway error. The first charge is valid; I've refunded the duplicate $49 today, and it should land back on your card in 5–7 business days.\n\nI've also added a note to prevent the retry from double-charging in future. Let me know if you don't see the refund by next week.\n\nBest,\nSam"},

    {"category": "billing", "subject": "Need a VAT invoice",
     "incoming": "Could you send me a proper invoice for our last payment that shows our company VAT number (GB123456789)? Our finance team needs it for the quarter.",
     "reference": "Hi there,\n\nHappy to help. I've regenerated your latest invoice with the VAT number GB123456789 added and emailed it to the billing contact on file. Going forward, any VAT number saved under Settings → Billing will appear automatically on every invoice.\n\nIf finance needs the previous quarters reissued the same way, just send me the invoice numbers and I'll sort them.\n\nBest,\nSam"},

    {"category": "billing", "subject": "Cancel before renewal",
     "incoming": "I don't want to renew — my plan renews on the 12th and I'd like to cancel before then so I'm not charged again.",
     "reference": "Hi,\n\nDone — I've switched your subscription to cancel at the end of the current period, so you won't be charged on the 12th and you'll keep full access until then. Nothing else is needed on your side.\n\nIf you change your mind before the 12th you can re-enable renewal from Settings → Billing, or just reply here. Mind sharing what prompted the cancellation? It genuinely helps us improve.\n\nBest,\nSam"},

    # ---------------- Refund ----------------
    {"category": "refund", "subject": "Refund for annual plan bought yesterday",
     "incoming": "I upgraded to the annual plan yesterday by mistake — I meant to stay monthly. Can I get a refund for the annual charge?",
     "reference": "Hi,\n\nNo problem at all. Since the annual upgrade was less than 24 hours ago and unused, I've fully refunded it ($468) and moved you back to the monthly plan at $49/mo. The refund takes 5–7 business days to appear.\n\nYou're all set on monthly now — no further action needed. Sorry for the mix-up!\n\nBest,\nSam"},

    {"category": "refund", "subject": "Unhappy, want money back",
     "incoming": "Honestly this hasn't worked for our team and I want a refund for this month. We barely used it.",
     "reference": "Hi,\n\nI'm sorry it didn't work out for your team — that's genuinely disappointing to hear. I've issued a full refund for this month's charge ($49), which will appear in 5–7 business days.\n\nBefore you go: if it was a specific gap (a missing integration, a workflow that didn't fit), I'd really like to hear it — sometimes there's a fix or a workaround we can set up. Either way, no pressure, and thanks for giving us a try.\n\nBest,\nSam"},

    # ---------------- Technical / bug ----------------
    {"category": "technical", "subject": "Emails not syncing to shared inbox",
     "incoming": "Since this morning, new emails aren't showing up in our shared inbox. We're on Chrome. Nothing changed on our end. This is blocking the whole support team.",
     "reference": "Hi,\n\nThanks for the details, and I understand how disruptive that is with the team blocked. This lines up with a sync delay on one of our mail-processing nodes that started around 08:00 UTC; our engineers are on it and new mail is already flowing again for most accounts.\n\nTwo quick things to get you unblocked now: (1) in the shared inbox, click Refresh (or reload the tab) to force a re-sync, and (2) confirm the Gmail connection under Settings → Connected Accounts still shows \"Active\". If it shows \"Reauthorise\", clicking it will pull the backlog through immediately.\n\nI'll email you the moment the node is fully cleared. If you're still not seeing new mail after a refresh, reply here and I'll escalate your account directly.\n\nBest,\nSam"},

    {"category": "technical", "subject": "Getting a 403 when assigning conversations",
     "incoming": "Every time I try to assign a conversation to a teammate I get an error that says '403 forbidden'. I'm an admin. Others on the team can assign fine.",
     "reference": "Hi,\n\nThanks — the fact that others can assign but you can't points to a permission scope on your specific seat rather than an outage. A 403 here almost always means the Gmail token for your account is missing the delegation scope.\n\nCould you try this: Settings → Connected Accounts → next to your account click \"Reauthorise\", and make sure you tick the \"Manage conversations\" permission on the Google consent screen. That re-grants the scope and clears the 403 in ~90% of cases.\n\nIf it still happens after re-authorising, send me a screenshot of the error and the rough time it occurred and I'll pull the server logs for your account.\n\nBest,\nSam"},

    {"category": "technical", "subject": "Notes feature is really slow",
     "incoming": "The internal notes on conversations take like 10 seconds to load lately. It's usable but annoying.",
     "reference": "Hi,\n\nAppreciate you reporting it — 10 seconds is definitely not normal, notes should be near-instant. A couple of things help us narrow it down: does it happen on every conversation or only ones with a long history, and are you seeing it in one browser or several?\n\nIn the meantime, a hard refresh (Cmd/Ctrl+Shift+R) clears a cached bundle that's been causing slow note loads for some users after our last update. If it persists, I'll get our team to profile it — mind sharing your rough location/region so we can check the nearest server?\n\nBest,\nSam"},

    # ---------------- How-to / product question ----------------
    {"category": "howto", "subject": "How do I set up an auto-reply for after hours?",
     "incoming": "We want customers who email us after 6pm to get an automatic acknowledgement. Is that possible and how?",
     "reference": "Hi,\n\nYes, that's built in. You'll want an Automation with a business-hours condition:\n\n1. Go to Settings → Automations → New Rule.\n2. Trigger: \"New conversation received\".\n3. Condition: \"Outside business hours\" (set your hours and timezone under Settings → Business Hours first).\n4. Action: \"Send reply\" and paste your acknowledgement template.\n\nSave it and it'll fire automatically on any email that arrives after 6pm. A tip: keep the message short and set expectations (\"we've got your email and will reply by the next business morning\"). Want me to share a template we've seen work well?\n\nBest,\nSam"},

    {"category": "howto", "subject": "Can two agents work the same inbox without collisions?",
     "incoming": "My worry with a shared inbox is two people replying to the same customer at once. How do you prevent that?",
     "reference": "Hi,\n\nGreat question — that's exactly the problem shared inboxes are meant to solve. Two things prevent collisions:\n\n1. Collision detection: if a teammate opens or starts typing on a conversation, you'll see a live indicator so you don't double-reply.\n2. Assignment: assign a conversation to one person and it's clearly \"owned\" by them; unassigned items stay in the shared queue.\n\nMost teams use assignment as the primary workflow and rely on collision detection as the safety net. Happy to walk your team through setting up assignment rules if that'd help.\n\nBest,\nSam"},

    # ---------------- Account / access ----------------
    {"category": "account", "subject": "Add 3 new team members",
     "incoming": "We just hired 3 support agents. Can you add them? Emails: rob@, priya@, and ken@ (all @brightloop.io). What happens to our billing?",
     "reference": "Hi,\n\nCongrats on the new hires! You can add them yourself in seconds: Settings → Team → Invite, then paste rob@, priya@ and ken@brightloop.io. They'll get an invite email to set a password.\n\nOn billing: your Growth plan is priced per active seat, so adding 3 agents adds 3 seats. We prorate, so you'll only be charged for the remaining days in this cycle, and the full amount applies from next renewal. If you'd like, I can send an exact proration estimate before you invite them.\n\nBest,\nSam"},

    {"category": "account", "subject": "Locked out — password reset not arriving",
     "incoming": "I've requested a password reset three times and no email is coming through. I need to get in today.",
     "reference": "Hi,\n\nSorry you're locked out — let's fix it fast. Reset emails not arriving is usually one of three things: they're landing in spam, the address has a typo, or a mail filter is catching them. Could you check spam/junk for a message from no-reply@ourdomain.com first?\n\nIf it's genuinely not there, I can trigger a reset link from our side and send it to a verified address on your account. Can you confirm the email you're trying to log in with? Once you do, I'll get you a working link within a few minutes.\n\nBest,\nSam"},

    # ---------------- Feature request ----------------
    {"category": "feature_request", "subject": "Slack notifications?",
     "incoming": "Would love to get a Slack ping when a high-priority conversation is assigned to me. Do you support that?",
     "reference": "Hi,\n\nYou can do this today. Under Settings → Integrations → Slack, connect your workspace, then create an Automation: Trigger \"Conversation assigned\", Condition \"Priority is High\", Action \"Send Slack notification\" to your channel or DM.\n\nIf you want it to ping the individual assignee rather than a shared channel, choose the \"notify assignee\" option in the Slack action. Let me know if you hit a snag connecting the workspace — the admin approval step in Slack trips a few people up.\n\nBest,\nSam"},

    {"category": "feature_request", "subject": "Bulk export of conversations",
     "incoming": "Is there a way to export all conversations from last quarter to CSV? We need it for an audit.",
     "reference": "Hi,\n\nYes — Analytics → Export lets you pull conversations for a date range to CSV, including status, assignee, tags and timestamps. Set the range to last quarter and choose \"Conversations (detailed)\".\n\nFor a compliance audit you may also want the full message bodies; that's a separate \"Full transcript export\" toggle on the same screen (it takes a few minutes to compile and emails you a download link). If the audit needs a specific field that's not in the export, tell me what it is and I'll check whether we can include it.\n\nBest,\nSam"},

    # ---------------- Complaint / escalation ----------------
    {"category": "complaint", "subject": "This is the third time I'm writing",
     "incoming": "This is the THIRD time I'm emailing about the same issue and no one has fixed it. Our reports still show the wrong numbers. I'm extremely frustrated.",
     "reference": "Hi,\n\nYou're right to be frustrated, and I'm sorry — three emails on the same issue is three too many, and that's on us. I don't want to send you round again, so I'm taking ownership of this personally.\n\nSo I can fix it rather than re-ask: can you confirm which report and what the numbers *should* say versus what they show? The moment I have that I'll reproduce it, loop in engineering with priority, and give you a concrete timeline the same day — not another holding reply. If it's easier, I'm happy to hop on a 15-minute call; just tell me a time.\n\nAgain, apologies for the run-around. I'll see this through.\n\nBest,\nSam"},

    {"category": "complaint", "subject": "Considering switching providers",
     "incoming": "We've had a lot of small issues and my team is talking about moving to a competitor. Give me a reason to stay.",
     "reference": "Hi,\n\nThank you for being straight with me — I'd rather hear this than have you leave quietly. A string of small issues adds up to a real loss of trust, and I don't want to hand-wave that away.\n\nHere's what I can concretely do: I'd like to set up a 30-minute review where we go through each issue you've hit, fix what's fixable on the spot, and put owners and dates on anything that isn't. I'll also connect you with our team directly so you're not starting from scratch each time. If it helps while we rebuild that trust, I can look at your renewal terms too.\n\nCan I grab 30 minutes with you this week? Tell me a couple of times and I'll work around them.\n\nBest,\nSam"},

    # ---------------- Onboarding / sales-ish ----------------
    {"category": "onboarding", "subject": "Just signed up — where do I start?",
     "incoming": "We just started the trial. What are the first things we should set up to get value quickly?",
     "reference": "Hi, and welcome!\n\nThe fastest path to value is usually these four, in order:\n\n1. Connect your shared mailbox (Settings → Connected Accounts) so real mail flows in.\n2. Invite your team (Settings → Team) — collaboration is where the payoff is.\n3. Set up assignment so every conversation has a clear owner.\n4. Add 2–3 saved templates for your most common replies.\n\nThat's usually enough to feel the difference within a day. I've also got a 20-minute onboarding call slot if you'd like a guided walkthrough — no sales pitch, just setup. Want me to send a link?\n\nBest,\nSam"},

    {"category": "onboarding", "subject": "Does it work with Outlook or only Gmail?",
     "incoming": "We're a Google Workspace shop now but might move to Outlook next year. Will we be stuck?",
     "reference": "Hi,\n\nToday the deepest integration is with Gmail / Google Workspace — that's where the in-inbox experience lives, so you're in the best-supported setup right now. Outlook support is more limited at the moment, so if you moved everything to Outlook next year, some of the in-mailbox features wouldn't carry over one-to-one.\n\nI don't want to oversell it: if an Outlook migration is firm, let's talk before you commit so I can tell you exactly what would and wouldn't work for your workflows. Happy to be candid about the trade-offs.\n\nBest,\nSam"},

    # ---------------- Data / privacy ----------------
    {"category": "privacy", "subject": "GDPR deletion request from a customer",
     "incoming": "One of our end-customers has asked us to delete all their personal data. Some of it is in conversations stored in your system. How do we handle that?",
     "reference": "Hi,\n\nThanks for handling this properly. You can action most of it yourself: search the customer's email address, open each conversation, and use \"Delete conversation\" — that removes the message content and their personal data from your workspace.\n\nFor a complete GDPR erasure (including from backups within the statutory window), send us a written deletion request naming the data subject's email from your admin address, and we'll process it against our systems and confirm in writing. As your processor, we're set up for exactly this. Want me to send the short request template so it has everything our compliance team needs?\n\nBest,\nSam"},

    # ---------------- Ambiguous / needs clarification ----------------
    {"category": "clarification", "subject": "it's broken",
     "incoming": "nothing works pls help",
     "reference": "Hi,\n\nI'm sorry you're hitting trouble — let's get it sorted. To point me in the right direction, could you tell me a little more:\n\n- What were you trying to do when it stopped working?\n- What do you see (an error message, a blank screen, something not loading)?\n- Is it happening for just you or others on your team too?\n\nEven a screenshot helps a lot. As soon as I have that I'll dig in straight away.\n\nBest,\nSam"},

    {"category": "clarification", "subject": "Question about the thing we discussed",
     "incoming": "Hi, following up on the thing from last week — can you go ahead with it?",
     "reference": "Hi,\n\nHappy to move forward — I just want to make sure I action the right thing. I'm not certain which request this refers to, and I don't want to assume. Could you remind me which item you mean (a rough date or the subject line of the earlier email is perfect)?\n\nOnce you confirm, I'll get straight on it.\n\nBest,\nSam"},

    # ---------------- Shipping/order (generic commerce, tests fact-carryover) ----------------
    {"category": "order", "subject": "Where is order #48213?",
     "incoming": "My order #48213 was supposed to arrive Tuesday and it's now Friday. Tracking hasn't updated in 3 days. Getting worried.",
     "reference": "Hi,\n\nSorry for the worry — a 3-day tracking gap on order #48213 isn't what we want. I checked and the parcel is in transit but stuck at a regional hub, which is why tracking went quiet; it hasn't been lost.\n\nHere's what I've done: I've opened a trace with the carrier and flagged #48213 for priority handling. If it doesn't move within 48 hours, I'll ship a replacement at no cost rather than leave you waiting. I'll update you either way by Monday. Thanks for your patience.\n\nBest,\nSam"},

    {"category": "order", "subject": "Wrong item received",
     "incoming": "I ordered the blue medium hoodie (order #51027) but received a red large. Need the right one before next weekend.",
     "reference": "Hi,\n\nSorry about that — a red large instead of your blue medium on order #51027 is our mistake to fix. I've placed a replacement for the blue medium today with expedited shipping so it reaches you before next weekend, at no extra cost.\n\nNo need to send the wrong one back in a hurry — I'll email a prepaid return label you can use whenever it's convenient. You'll get tracking for the replacement within 24 hours.\n\nBest,\nSam"},
]

# Parametrised augmentation: same *shapes* as real tickets, varied specifics.
# Clearly labelled synthetic-augmented so we don't overstate dataset size.
AUG_TEMPLATES = [
    {"category": "billing",
     "incoming": "Hi, I was charged {amt} on {date} but I thought my plan was {plan}. Can you explain the charge on account {acct}?",
     "reference": "Hi,\n\nThanks for checking in. Looking at {acct}, the {amt} charge on {date} is your {plan} subscription renewing for the period — it's a valid, expected charge, not a duplicate. I've emailed the itemised invoice so you can see the breakdown.\n\nIf the plan itself is more than you need, I'm happy to walk through options that lower the monthly cost. Just say the word.\n\nBest,\nSam"},
    {"category": "order",
     "incoming": "Order {oid} still hasn't arrived and it's past the {date} estimate. Can you check?",
     "reference": "Hi,\n\nSorry it's late — I've looked into order {oid} and it's past the {date} estimate because of a carrier delay at a sorting hub. It's still on its way, not lost.\n\nI've flagged {oid} for priority handling and opened a trace. If it hasn't moved in 48 hours I'll send a replacement free of charge. I'll keep you posted.\n\nBest,\nSam"},
    {"category": "account",
     "incoming": "Can you remove {name} from our team? They've left the company. Account: {acct}.",
     "reference": "Hi,\n\nDone — I've removed {name} from the team on {acct} and revoked their access immediately, so they can no longer see any conversations. Since seats are billed per active user, your next invoice will drop by one seat, prorated for the rest of this cycle.\n\nIf you'd like me to reassign {name}'s open conversations to someone specific, just tell me who.\n\nBest,\nSam"},
]

AUG_VALUES = {
    "amt": ["$29", "$49", "$99", "$149"],
    "date": ["the 2nd", "the 9th", "May 14", "June 1"],
    "plan": ["Starter", "Growth", "Pro", "the monthly Growth plan"],
    "acct": ["acme@corp.com", "hello@finch.io", "team@nova.co", "ops@brightloop.io"],
    "oid": ["#40912", "#55130", "#61288", "#72004"],
    "name": ["Alex Morgan", "Priya Shah", "Chris Doyle", "Lena Park"],
}


def _augment(n_per_template: int = 4):
    rows = []
    for t in AUG_TEMPLATES:
        for i in range(n_per_template):
            fields = {}
            for key, opts in AUG_VALUES.items():
                fields[key] = opts[i % len(opts)]
            rows.append({
                "category": t["category"],
                "subject": "(augmented) " + t["category"],
                "incoming": t["incoming"].format(**fields),
                "reference": t["reference"].format(**fields),
                "source": "synthetic_augmented",
            })
    return rows


def build(augment: bool = False):
    rows = []
    for r in CORE:
        r = dict(r)
        r.setdefault("source", "hand_authored")
        rows.append(r)
    if augment:
        rows.extend(_augment())

    # Deterministic id + split (80/20) via stable hash of the incoming text.
    out = []
    for r in rows:
        h = hashlib.sha1(r["incoming"].encode()).hexdigest()
        r["id"] = h[:10]
        # ~20% to test, deterministically
        r["split"] = "test" if int(h[:8], 16) % 5 == 0 else "train"
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true", help="add parametrised synthetic rows")
    ap.add_argument("--out", default="data/emails.jsonl")
    args = ap.parse_args()

    rows = build(augment=args.augment)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = len(rows) - n_train
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"Wrote {len(rows)} records to {args.out}  (train={n_train}, test={n_test})")
    print("Categories:", json.dumps(cats, indent=None))


if __name__ == "__main__":
    main()
