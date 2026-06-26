import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import umap
import hdbscan

# ======================================================================
# LOAD DATA
# ======================================================================

df = pd.read_csv("csv_data/a.csv")

# ======================================================================
# STEP 1 — CLEAN TEXT
#
# CHANGE FROM PREVIOUS VERSION:
# We no longer substitute {product_purchased} with the real product
# name before clustering. Clustering is about ISSUE TYPE, not which
# product was bought — injecting "dell xps" / "gopro hero" / etc. into
# the embedded text adds product-specific vocabulary that has nothing
# to do with the symptom, and it's different for every product, so it
# pulls tickets apart along a "which product" axis instead of a
# "what's wrong" axis. This was quietly lowering silhouette score.
#
# Because we skip the substitution, the literal "{product_purchased}"
# placeholder is matched and removed directly by the wrapper noise
# pattern below — no separate "strip the generic shell" regex needed
# anymore (that regex existed only to clean up residue left behind by
# the substitution, which no longer happens).
# ======================================================================

_NOISE_PATTERNS_RAW = [
    # Outer wrapper shell — matches the literal placeholder directly
    r"I[\u2019']?m\s+having\s+an\s+issue\s+with\s+the\s+\{product[_ ]?purchased\}\s*[.,]?\s*(?:Please\s+assist)?\s*[.,]?",
    r"Please\s+assist\s*[.,]?",

    # E-commerce / forum noise
    r"Your\s+billing\s+zip\s+code\s+is[^.\n]*[.?\n]?",
    r"We\s+appreciate\s+that\s+you\s+have\s+requested\s+a\s+website\s+address[^.\n]*[.?\n]?",
    r"Please\s+double\s+check\s+your\s+email\s+address[^.\n]*[.?\n]?",
    r"If\s+you\s+need\s+to\s+change\s+an\s+existing\s+product[^.\n]*[.?\n]?",
    r"The\s+seller\s+is\s+not\s+responsible\s+for\s+any\s+damages[^.\n]*[.?\n]?",
    r"Please\s+have\s+the\s+game\s+in\s+good\s+condition[^.\n]*[.?\n]?",
    r"Note\s*:\s*The\s+seller[^.\n]*[.?\n]?",
    r"If\s+you\s+have\s+a\s+problem\s+you[\u2019']?re\s+interested\s+in[^.\n]*[.?\n]?",
    r"I\s+will\s+contact\s+all\s+my\s+suppliers[^.\n]*[.?\n]?",
    r"Please\s+try\s+and\s+find\s+out\s+whether\s+their\s+inventory[^.\n]*[.?\n]?",
    r"Product\s+Search\s*:[^.\n]*[.?\n]?",
    r"Product\s+Name\s*:\s*\S+[^.\n]*[.?\n]?",
    r"Join\s+Date\s*:[^.\n]*[.?\n]?",
    r"Report\s+Feedback\s+Customer\s+Service[^.\n]*[.?\n]?",
    r"I\s+can[\u2019']?t\s+find\s+the\s+[\u2018'\"]?Product_IP[\u2018'\"]?\s+field[^.\n]*[.?\n]?",
    r"Solution\s+\d+\s*",
    r"To\s+remove\s+the\s+new\s+\{product[_ ]?purchased\}[^.\n]*[.?\n]?",

    # Boilerplate closing actions
    r"I[\u2019']?ve\s+(?:already\s+)?contacted\s+customer\s+support\s+multiple\s+times[^.\n]*[.?\n]?",
    r"I[\u2019']?ve\s+reviewed\s+the\s+troubleshooting\s+steps\s+on\s+the\s+official\s+support\s+website[^.\n]*[.?\n]?",
    r"I[\u2019']?ve\s+(?:tried|followed)\s+(?:the\s+)?troubleshooting\s+steps\s+mentioned\s+in\s+the\s+user\s+manual[^.\n]*[.?\n]?",
    r"I[\u2019']?ve\s+performed\s+a\s+factory\s+reset[^.\n]*[.?\n]?",
    r"I[\u2019']?ve\s+tried\s+clearing\s+the\s+cache[^.\n]*[.?\n]?",
    r"I[\u2019']?ve\s+checked\s+for\s+any\s+available\s+software\s+updates[^.\n]*[.?\n]?",
    r"I[\u2019']?ve\s+checked\s+the\s+device\s+settings[^.\n]*[.?\n]?",
    r"hoping\s+it\s+would\s+resolve\s+the\s+problem[^.\n]*[.?\n]?",
    r"but\s+it\s+didn[\u2019']?t\s+help[^.\n]*[.?\n]?",

    # Remaining placeholder tokens
    r"\{error[_ ]?message\}",
    r"\{product[_ ]?purchased\}",
]
_NOISE_RE_RAW = re.compile(
    "|".join(_NOISE_PATTERNS_RAW),
    flags=re.IGNORECASE | re.DOTALL,
)

# Bare token form — safety net in case a template fragment elsewhere
# in the ticket left "product purchased" without braces
_PROD_TOKEN_RE = re.compile(
    r"\bproduct[_\s]?purch(?:ased)?\b",
    flags=re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = str(text)

    # Step 1 — remove noise blocks (placeholder included, no substitution)
    text = _NOISE_RE_RAW.sub(" ", text)

    # Step 2 — lowercase
    text = text.lower()

    # Step 3 — remove any bare leftover placeholder token
    text = _PROD_TOKEN_RE.sub(" ", text)

    # Step 4 — standard sanitisation
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["Ticket_clean"] = df["Ticket Description"].apply(clean_text)

# ======================================================================
# STEP 2 — SENTENCE DEDUP
# Remove duplicate sentences within a single ticket
# ======================================================================

def dedup_sentences(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 5]
    seen, unique = set(), []
    for s in sentences:
        key = re.sub(r"\s+", " ", s)[:100]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return " ".join(unique)


df["Ticket_clean"] = df["Ticket_clean"].apply(dedup_sentences)

# ======================================================================
# STEP 3 — ISSUE EXTRACTION
#
# CHANGE FROM PREVIOUS VERSION:
# Tickets are built from 2-3 stitched issue sentences. The old code
# joined EVERY sentence that matched a keyword, which often blended
# two different problems (e.g. a battery sentence + a wifi sentence)
# into one embedding. That structurally caps how tight any cluster can
# get, no matter how UMAP/HDBSCAN are tuned.
#
# Fix: keep only the SINGLE sentence with the strongest issue-keyword
# signal per ticket (most keyword hits, longest as tiebreak). One
# ticket -> one issue, every time.
#
# Keyword list also trimmed: removed generic severity/impact words
# (persists, unresolved, affecting, productivity, hindering, security,
# safe, ensure, intermittent, unexpectedly, strange, functioning).
# Those describe how bad/long-running a problem is, not what KIND of
# problem it is — they fire across every category and just add noisy
# extra sentence matches.
# ======================================================================

ISSUE_KEYWORDS = [
    # auth
    "password", "login", "credentials", "locked", "unlock",
    "access", "invalid", "forgot", "forgotten",
    # connectivity
    "network", "wifi", "wi-fi", "connecting", "connect", "internet",
    "disconnect", "disconnecting", "bluetooth",
    # software / firmware
    "firmware", "software", "update", "install", "crash", "crashes",
    "freeze", "freezes", "glitch", "reboot", "restart", "hang",
    "slow", "lag", "bug", "error", "version",
    # hardware / physical
    "battery", "charging", "charger", "drain", "screen",
    "display", "flickering", "flicker", "crack", "noise", "noises",
    "overheat", "turning on", "not turning", "hardware",
    "repair", "replacement",
    # data
    "deleted", "recover", "lost", "missing", "backup",
    "corrupted", "disappeared", "data loss",
    # general failure
    "not working", "unable", "cannot", "doesn", "won",
    "fail", "fails", "failing",
]


def extract_issue(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 5]
    scored = []
    for s in sentences:
        hits = sum(1 for kw in ISSUE_KEYWORDS if kw in s)
        if hits > 0:
            scored.append((hits, len(s), s))
    if not scored:
        return text
    # most keyword hits wins; longer sentence breaks ties
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


df["Issue_Only"] = df["Ticket_clean"].apply(extract_issue)

# ======================================================================
# STEP 4 — STOPWORDS
# These are stripped before embedding AND before TF keyword extraction.
# Nothing in this list should ever appear as a cluster label.
#
# Brand-name stopwords kept as a harmless safety net even though
# product names are no longer injected by clean_text — in case any
# brand name appears organically in the free-text part of a ticket.
# ======================================================================

BOILERPLATE_WORDS = {
    # template residue
    "issue", "problem", "please", "assist",
    "customer", "ticket", "user", "help", "support",
    "having", "facing",
    # greetings / sign-offs
    "thanks", "thank", "hi", "hello", "dear", "regards", "sincerely",
    "best", "hope", "know",
    # contraction residue
    "ve", "re", "ll", "don", "isn", "aren", "wasn", "weren",
    "hasn", "haven", "hadn", "doesn", "didn", "couldn", "wouldn",
    "shouldn", "ive", "im", "id",
    # filler adverbs / conjunctions
    "also", "still", "even", "however", "though", "although",
    "really", "actually", "basically", "just", "like", "need",
    "want", "make", "get", "got", "getting", "going",
    # generic verbs
    "work", "working", "works", "worked",
    "use", "uses", "using",
    "tried", "try", "check", "checked",
    "fix", "fixed", "resolve", "resolved",
    "contact", "contacted", "restart", "restarted", "reset",
    "start", "started", "stop", "stopped",
    # vague nouns
    "device", "unit", "system", "service",
    "thing", "time", "way", "day", "week", "month",
    "new", "old", "good", "bad", "same", "different",
    # words that describe actions TAKEN, not the issue type
    "recently", "already", "multiple", "original",
    # words that polluted cluster labels in previous run
    "fine", "acts", "times", "happen", "come", "came",
    "sure", "possible", "soon", "far", "luck", "followed", "online",
    "properly", "related", "require", "worried",
    # generic modal / question words
    "what", "how", "when", "where", "why", "which", "who",
    "can", "could", "would", "should", "may", "might",
    # generic intent words that bleed across all clusters
    "provide", "guide", "steps", "option", "action",
    "perform", "desired",
    # product brand names (metadata, not issue signal — safety net)
    "iphone", "samsung", "apple", "google", "sony", "lg", "dell",
    "microsoft", "amazon", "fitbit", "gopro", "nikon", "canon",
    "bose", "dyson", "nest", "nintendo", "philips", "garmin",
    "roomba", "playstation", "xbox", "autodesk", "autocad",
}

clustering_stopwords = set(ENGLISH_STOP_WORDS) | BOILERPLATE_WORDS


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in clustering_stopwords)


df["Issue_For_Embedding"] = df["Issue_Only"].apply(remove_stopwords)
df["Issue_For_TF"]        = df["Issue_For_Embedding"]

# ── Sanity check ──────────────────────────────────────────────────────
print("\n=== SAMPLE CLEANED TICKETS ===")
for i in [1, 2, 6, 9, 13]:
    raw   = df["Ticket Description"].iloc[i][:130]
    clean = df["Issue_For_Embedding"].iloc[i]
    print(f"  RAW   [{i}]: {raw!r}")
    print(f"  CLEAN [{i}]: {clean!r}\n")
print("=" * 70)

# ======================================================================
# STEP 5 — BGE EMBEDDINGS
# ======================================================================

print("\nLoading BGE Model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

print("Generating embeddings...")
X = model.encode(df["Issue_For_Embedding"].tolist(), show_progress_bar=True)
print("Embedding Shape :", X.shape)

np.save("ticket_embeddings.npy", X)
print("Embeddings saved -> ticket_embeddings.npy")

# ======================================================================
# STEP 6 — UMAP
# ======================================================================

print("\nRunning UMAP...")
reducer = umap.UMAP(
    n_neighbors=20,
    n_components=20,
    metric="cosine",
    min_dist=0.0,       # tight clusters → better HDBSCAN separation
    random_state=42,
)
X_umap = reducer.fit_transform(X)
print("Reduced Shape   :", X_umap.shape)

# ======================================================================
# STEP 7 — HDBSCAN
# ======================================================================

SEP = "=" * 90

print(f"\n\n{SEP}")
print("STEP 7 — HDBSCAN CLUSTERING")
print(SEP)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=5,
    metric="euclidean",
    cluster_selection_epsilon=0.5,
    cluster_selection_method="eom",
    prediction_data=True,
)
labels = clusterer.fit_predict(X_umap)
df["cluster"]             = labels
df["cluster_probability"] = clusterer.probabilities_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"Clusters Found : {n_clusters}")
print(f"Noise Tickets  : {n_noise}  ({100*n_noise/len(df):.2f}%)")

# ======================================================================
# HELPERS
# ======================================================================

def get_unique_tickets(docs, top_n=5):
    seen, unique = set(), []
    for doc in docs:
        key = re.sub(r"\s+", " ", doc.strip())[:120]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
        if len(unique) >= top_n:
            break
    return unique

# ======================================================================
# STEP 8 — CENTROID KEYWORD EXTRACTION + REPRESENTATIVE TICKETS
# ======================================================================

TOP_K_CENTRAL = 100

cluster_summary  = []
cluster_labels   = {}
cluster_details  = {}

cluster_sizes = (
    df[df["cluster"] != -1]
    .groupby("cluster")
    .size()
    .sort_values(ascending=False)
)

print(f"\n\n{SEP}")
print("STEP 8 — CENTROID LABELLING")
print(SEP)

for cluster_id, cluster_size in cluster_sizes.items():

    cluster_idx          = df[df["cluster"] == cluster_id].index.tolist()
    cluster_embs         = X[cluster_idx]
    cluster_docs_display = df.loc[cluster_idx, "Issue_Only"].tolist()
    cluster_docs_tf      = df.loc[cluster_idx, "Issue_For_TF"].tolist()
    avg_prob             = df.loc[cluster_idx, "cluster_probability"].mean()

    centroid  = cluster_embs.mean(axis=0, keepdims=True)
    distances = pairwise_distances(centroid, cluster_embs, metric="cosine")[0]

    k         = min(TOP_K_CENTRAL, len(cluster_docs_display))
    top_k_idx = np.argsort(distances)[:k]

    central_docs_tf = [cluster_docs_tf[i] for i in top_k_idx]
    vectorizer = CountVectorizer(
        stop_words=list(clustering_stopwords),
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2,
    )

    try:
        X_tf      = vectorizer.fit_transform(central_docs_tf)
        terms     = vectorizer.get_feature_names_out()
        scores    = X_tf.sum(axis=0).A1
        top_words = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:10]
    except ValueError:
        top_words = [("(no terms)", 0)]

    top_word = top_words[0][0]
    cluster_labels[cluster_id] = top_word

    central_docs_display = [cluster_docs_display[i] for i in top_k_idx]
    display_tickets      = get_unique_tickets(central_docs_display, top_n=5)

    print(f"\n{'─'*90}")
    print(f"  Cluster {cluster_id:>3}  |  {cluster_size:>4} tickets  |  "
          f"Avg Confidence: {avg_prob:.3f}")
    print(f"  Auto Label  : {top_word}")
    print(f"  Keywords    : {' | '.join(f'{w}({int(s)})' for w,s in top_words[:6])}")
    print(f"  Representative tickets:")
    for t in display_tickets:
        print(f"    • {t}")

    cluster_details[cluster_id] = {
        "size"    : cluster_size,
        "label"   : top_word,
        "keywords": top_words,
        "tickets" : display_tickets,
        "avg_prob": avg_prob,
    }

    cluster_summary.append({
        "cluster"        : cluster_id,
        "ticket_count"   : cluster_size,
        "top_word"       : top_word,
        "tf_keywords"    : ", ".join(w for w, _ in top_words),
        "avg_probability": avg_prob,
    })

# ======================================================================
# AUTO LABEL (keyword fallback)
# ======================================================================

df["cluster_label"] = df["cluster"].map(cluster_labels)
df.loc[df["cluster"] == -1, "cluster_label"] = "noise"

# ======================================================================
# MANUAL LABELS
# Cluster IDs will be different from your previous run since the input
# text changed — re-read the representative tickets above and re-fill
# this after you've picked a final UMAP/HDBSCAN config.
# ======================================================================

MANUAL_LABELS = {
    # ── Software / Firmware ─────────────────────────────────
    28: "Software Issue",          # post-update issues, no other changes
     0: "Firmware Update Error",   # firmware update caused the problem
    14: "Software Issue",          # software glitch, frequent freezes
    25: "Software Issue",          # software bug, app crash on specific action
    16: "Application Crash",       # bug in app causing data loss + errors
     1: "Application Issue",       # issue occurs consistently on specific feature
    # ── Hardware ────────────────────────────────────────────
    11: "Hardware Issue",          # screen flickering, can't use device
    17: "Hardware Issue",          # may require repair or replacement
    13: "Hardware Issue",          # strange noises, not functioning properly
    10: "Hardware Issue",          # widespread across multiple same-model devices
     4: "Hardware Issue",          # tried cables/adapters/peripherals, persists
     5: "Battery Life Issue",      # sudden decrease in battery life
    12: "Charging Issue",          # original charger not charging properly
    # ── Network / Connectivity ──────────────────────────────
     9: "Network Issue",           # unstable internet, keeps disconnecting
     2: "Network Issue",           # Wi-Fi not detected, other devices fine
    18: "Network Issue",           # fails to connect to any network after setup
    # ── Account / Access ────────────────────────────────────
    23: "Account Recovery",        # forgotten password, reset not working
    22: "Account Locked",          # account locked, login issues
    21: "Account Access Issue",    # invalid credentials despite correct info
    # ── Data & Security ─────────────────────────────────────
    20: "Data Recovery",           # accidental deletion / data loss
     8: "Data Recovery",           # data loss + error message (mixed cluster)
    26: "Security Issue",          # concerned about device/data security
    # ── Error Messages & Config ─────────────────────────────
    15: "Error Message",           # peculiar error message on screen
    19: "Configuration Issue",     # tried different settings, issue persists
    27: "Guidance Request",        # can't find option, needs step-by-step help
    # ── Productivity / Urgency ──────────────────────────────
     6: "Urgent Support Request",  # affecting work and productivity
     7: "Productivity Issue",      # relies on device for daily tasks
    24: "Intermittent Issue",      # works sometimes, acts up unexpectedly
    # ── Noise ───────────────────────────────────────────────
     3: "Noise",                   # empty/incomplete tickets, no issue content
}

df["cluster_label"] = df["cluster"].map(MANUAL_LABELS).fillna(df["cluster_label"])
df.loc[df["cluster"] == -1, "cluster_label"] = "noise"

# ======================================================================
# STEP 9 — SILHOUETTE SCORE
# ======================================================================

mask      = labels != -1
sil_score = None
if len(set(labels[mask])) > 1:
    sil_score = silhouette_score(X_umap[mask], labels[mask])
    print(f"\nSilhouette Score: {round(sil_score, 4)}")

# ======================================================================
# STEP 10 — VALIDATION REPORT
# ======================================================================

def quality_label(p):
    if p >= 0.85: return "Excellent"
    if p >= 0.70: return "Good"
    if p >= 0.60: return "Fair"
    return "Poor"

validation_rows = []
for cluster_id, detail in sorted(cluster_details.items(),
                                 key=lambda x: x[1]["size"], reverse=True):
    avg_p  = detail["avg_prob"]
    validation_rows.append({
        "cluster"        : cluster_id,
        "final_label"    : MANUAL_LABELS.get(cluster_id, detail["label"]),
        "auto_keyword"   : detail["label"],
        "tickets"        : detail["size"],
        "avg_probability": round(avg_p, 3),
        "quality"        : quality_label(avg_p),
        "inspection"     : "REVIEW" if avg_p < 0.60 else "PASS",
    })

validation_df = pd.DataFrame(validation_rows)

print(f"\n\n{SEP}")
print("STEP 10 — VALIDATION SUMMARY")
print(SEP)
print(f"  Silhouette Score : {sil_score}")
print(f"  Noise Tickets    : {n_noise}  ({100*n_noise/len(df):.2f}%)")
print()
print(validation_df.to_string(index=False))

# ======================================================================
# STEP 11 — CONFIDENCE FILTER → TRAINING SET
# ======================================================================

CONFIDENCE_THRESHOLD = 0.60

training_df = (
    df[
        (df["cluster"] != -1) &
        (df["cluster_probability"] >= CONFIDENCE_THRESHOLD)
    ][[
        "Tech_City",
        "Ticket Description",
        "Issue_Only",
        "Priority",  
        "cluster_label",
        "cluster_probability",
    ]]
    .copy()
    .reset_index(drop=True)
)

training_indices = df[
    (df["cluster"] != -1) &
    (df["cluster_probability"] >= CONFIDENCE_THRESHOLD)
].index.to_numpy()

np.save("training_indices.npy", training_indices)

print(f"\nPre-filter rows  : {(df['cluster'] != -1).sum()}")
print(f"Post-filter rows : {len(training_df)}  (prob >= {CONFIDENCE_THRESHOLD})")
print(f"\nLabel distribution:\n{training_df['cluster_label'].value_counts().to_string()}")

# ======================================================================
# HOTSPOTS + NOISE
# ======================================================================

hotspots = (
    df[df["cluster"] != -1]
    .groupby(["cluster", "cluster_label", "Tech_City"])
    .size()
    .reset_index(name="ticket_count")
    .sort_values(by="ticket_count", ascending=False)
)

noise_tickets = df[df["cluster"] == -1].copy()
print(f"\nNoise Tickets: {len(noise_tickets)}")

# ======================================================================
# TXT REPORT
# ======================================================================

def write_txt_report(path, cluster_details, hotspots, noise_tickets, sil_score):
    S1 = "=" * 90
    S2 = "-" * 90
    S3 = "·" * 70

    with open(path, "w", encoding="utf-8") as f:
        f.write(S1 + "\n")
        f.write("TICKET CLUSTER ANALYSIS REPORT\n")
        f.write(S1 + "\n")
        f.write(f"Total Clusters   : {len(cluster_details)}\n")
        f.write(f"Noise Tickets    : {len(noise_tickets)}\n")
        if sil_score is not None:
            f.write(f"Silhouette Score : {round(sil_score, 4)}\n")
        f.write("\n")

        for cluster_id, detail in sorted(cluster_details.items(),
                                         key=lambda x: x[1]["size"], reverse=True):
            label = MANUAL_LABELS.get(cluster_id, detail["label"])
            f.write(S1 + "\n")
            f.write(
                f"CLUSTER {cluster_id:>3}  |  Tickets: {detail['size']:>5}  |  "
                f"Label: {label.upper()}  |  Avg Prob: {detail['avg_prob']:.3f}\n"
            )
            f.write(S1 + "\n")
            f.write("KEYWORDS:\n")
            for rank, (word, score) in enumerate(detail["keywords"], 1):
                f.write(f"  {rank:>2}. {word:<40}  {int(score)}\n")
            f.write(f"\nREPRESENTATIVE TICKETS:\n{S2}\n")
            for t_idx, ticket in enumerate(detail["tickets"], 1):
                f.write(f"[{t_idx}] {ticket}\n{S3}\n")
            f.write("\n")

        f.write(f"{S1}\nHOTSPOTS\n{S1}\n")
        for _, row in hotspots.iterrows():
            f.write(
                f"  Cluster {int(row['cluster']):>3}  |  "
                f"{row['cluster_label']:<35}  |  "
                f"Tickets: {int(row['ticket_count'])}\n"
            )

        f.write(f"\n{S1}\nNOISE TICKETS (total: {len(noise_tickets)})\n{S1}\n")
        for i, (_, row) in enumerate(noise_tickets.iterrows(), 1):
            f.write(f"[{i}] {str(row.get('Issue_Only', ''))}\n{S3}\n")

    print(f"Cluster report saved -> {path}")


write_txt_report("cluster_report.txt", cluster_details, hotspots, noise_tickets, sil_score)

# ======================================================================
# EXPORT
# ======================================================================

summary_df  = pd.DataFrame(cluster_summary)
df_filtered = df[
    (df["cluster"] != -1) &
    (df["cluster_probability"] >= CONFIDENCE_THRESHOLD)
].copy()

summary_df.to_csv("cluster_summary.csv",                index=False)
hotspots.to_csv("cluster_hotspots.csv",                 index=False)
df.to_csv("clustered_tickets.csv",                      index=False)
df_filtered.to_csv("filtered_clustered_tickets.csv",    index=False)
noise_tickets.to_csv("noise_tickets.csv",               index=False)
training_df.to_csv("ticket_training_dataset.csv",       index=False)
validation_df.to_csv("cluster_semantic_validation.csv", index=False)

with pd.ExcelWriter("ticket_clustering_report.xlsx") as writer:
    df.to_excel(writer,             sheet_name="Clustered Tickets",   index=False)
    summary_df.to_excel(writer,     sheet_name="Cluster Summary",     index=False)
    hotspots.to_excel(writer,       sheet_name="Hotspots",            index=False)
    noise_tickets.to_excel(writer,  sheet_name="Noise Tickets",       index=False)
    training_df.to_excel(writer,    sheet_name="Training Dataset",    index=False)
    validation_df.to_excel(writer,  sheet_name="Validation Report",   index=False)

training_df.to_excel("ticket_training_dataset.xlsx", index=False)

print(f"\n{SEP}")
print("PIPELINE COMPLETE")
print(SEP)
print(f"  Total Tickets          : {len(df)}")
print(f"  Clusters Found         : {n_clusters}")
print(f"  Noise Tickets          : {n_noise}  ({100*n_noise/len(df):.2f}%)")
print(f"  Silhouette Score       : {sil_score}")
print(f"  Final Training Samples : {len(training_df)}")