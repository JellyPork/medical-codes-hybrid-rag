import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
  import ollama
except ImportError:
  ollama = None


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / 'data' / 'medical_codes_sample.jsonl'
DEFAULT_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')
DEFAULT_CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF')


@dataclass
class MedicalCodeDocument:
  code: str
  title: str
  description: str
  synonyms: list[str]
  source: str = ''

  @property
  def text(self):
    parts = [self.code, self.title, self.description, ' '.join(self.synonyms)]
    return ' | '.join(part for part in parts if part)


@dataclass
class IndexedDocument:
  document: MedicalCodeDocument
  tokens: Counter
  length: int
  embedding: list[float] | None


def normalize_text(value):
  value = unicodedata.normalize('NFKD', value)
  value = ''.join(ch for ch in value if not unicodedata.combining(ch))
  return value.lower()


def normalize_code(value):
  return re.sub(r'[^a-z0-9]', '', normalize_text(value))


def tokenize(value):
  return re.findall(r'[a-z0-9]+', normalize_text(value))


def cosine_similarity(left, right):
  if not left or not right:
    return 0.0

  dot_product = sum(a * b for a, b in zip(left, right))
  left_norm = math.sqrt(sum(value * value for value in left))
  right_norm = math.sqrt(sum(value * value for value in right))
  if not left_norm or not right_norm:
    return 0.0
  return dot_product / (left_norm * right_norm)


def min_max_normalize(values):
  if not values:
    return []

  minimum = min(values)
  maximum = max(values)
  if math.isclose(minimum, maximum):
    return [1.0 if maximum > 0 else 0.0 for _ in values]

  return [(value - minimum) / (maximum - minimum) for value in values]


def load_documents(data_path):
  documents = []
  with data_path.open('r', encoding='utf-8') as handle:
    for line_number, line in enumerate(handle, start=1):
      line = line.strip()
      if not line:
        continue

      payload = json.loads(line)
      code = str(payload.get('code', '')).strip()
      title = str(payload.get('title', '')).strip()
      description = str(payload.get('description', '')).strip()
      synonyms = payload.get('synonyms', [])
      source = str(payload.get('source', '')).strip()

      if not code or not title or not description:
        raise ValueError(f'Invalid record on line {line_number}: code, title, and description are required.')

      if not isinstance(synonyms, list):
        raise ValueError(f'Invalid record on line {line_number}: synonyms must be a list.')

      documents.append(
        MedicalCodeDocument(
          code=code,
          title=title,
          description=description,
          synonyms=[str(item).strip() for item in synonyms if str(item).strip()],
          source=source,
        )
      )

  if not documents:
    raise ValueError(f'No documents found in {data_path}')

  return documents


def build_idf(indexed_documents):
  document_frequency = Counter()
  for indexed_document in indexed_documents:
    document_frequency.update(indexed_document.tokens.keys())

  total_documents = len(indexed_documents)
  return {
    token: math.log((1 + total_documents) / (1 + frequency)) + 1.0
    for token, frequency in document_frequency.items()
  }


def embed_text(text, model):
  if ollama is None:
    return None

  response = ollama.embed(model=model, input=text)
  return response['embeddings'][0]


def build_index(documents, embed_model, enable_semantic=True):
  indexed_documents = []
  semantic_enabled = enable_semantic and ollama is not None

  for document in documents:
    tokens = Counter(tokenize(document.text))
    embedding = None
    if semantic_enabled:
      try:
        embedding = embed_text(document.text, embed_model)
      except Exception as error:
        semantic_enabled = False
        print(f'Semantic retrieval disabled: {error}', file=sys.stderr)
        indexed_documents = [
          IndexedDocument(
            document=item.document,
            tokens=item.tokens,
            length=item.length,
            embedding=None,
          )
          for item in indexed_documents
        ]
        embedding = None

    indexed_documents.append(
      IndexedDocument(
        document=document,
        tokens=tokens,
        length=sum(tokens.values()),
        embedding=embedding,
      )
    )

  return indexed_documents, build_idf(indexed_documents), semantic_enabled


def lexical_score(query_tokens, indexed_document, idf, normalized_query_code):
  if not query_tokens:
    return 0.0

  score = 0.0
  for token, query_count in query_tokens.items():
    score += query_count * indexed_document.tokens.get(token, 0) * idf.get(token, 1.0)

  length_penalty = 1.0 + math.log(indexed_document.length + 1)
  score /= length_penalty

  document_code = normalize_code(indexed_document.document.code)
  if document_code and document_code in normalized_query_code:
    score += 3.0

  return score


def retrieve(query, indexed_documents, idf, embed_model, top_n=5, keyword_weight=0.45, semantic_weight=0.55, semantic_enabled=True):
  query_tokens = Counter(tokenize(query))
  normalized_query_code = normalize_code(query)
  lexical_scores = [lexical_score(query_tokens, item, idf, normalized_query_code) for item in indexed_documents]
  lexical_scores = min_max_normalize(lexical_scores)

  semantic_scores = [0.0 for _ in indexed_documents]
  if semantic_enabled:
    try:
      query_embedding = embed_text(query, embed_model)
      semantic_scores = [cosine_similarity(query_embedding, item.embedding) for item in indexed_documents]
      semantic_scores = min_max_normalize(semantic_scores)
    except Exception as error:
      semantic_enabled = False
      print(f'Semantic query scoring disabled: {error}', file=sys.stderr)

  active_keyword_weight = keyword_weight
  active_semantic_weight = semantic_weight if semantic_enabled else 0.0
  total_weight = active_keyword_weight + active_semantic_weight
  if total_weight:
    active_keyword_weight /= total_weight
    active_semantic_weight /= total_weight

  results = []
  for indexed_document, lexical, semantic in zip(indexed_documents, lexical_scores, semantic_scores):
    hybrid = (active_keyword_weight * lexical) + (active_semantic_weight * semantic)
    results.append({
      'document': indexed_document.document,
      'keyword_score': lexical,
      'semantic_score': semantic,
      'hybrid_score': hybrid,
    })

  results.sort(key=lambda item: item['hybrid_score'], reverse=True)
  return results[:top_n], semantic_enabled


def print_results(results):
  for rank, result in enumerate(results, start=1):
    document = result['document']
    print(f'{rank}. {document.code} - {document.title}')
    print(f'   hybrid={result["hybrid_score"]:.3f} keyword={result["keyword_score"]:.3f} semantic={result["semantic_score"]:.3f}')
    print(f'   {document.description}')
    if document.synonyms:
      print(f'   synonyms: {", ".join(document.synonyms)}')
    if document.source:
      print(f'   source: {document.source}')


def answer_query(query, results, chat_model):
  if ollama is None:
    print('\nGeneration skipped because the ollama Python package is not installed.')
    return

  context_lines = []
  for result in results:
    document = result['document']
    context_lines.append(
      f'Code: {document.code}\nTitle: {document.title}\nDescription: {document.description}\nSynonyms: {", ".join(document.synonyms) or "none"}'
    )

  system_prompt = (
    'You answer questions about medical coding using only the provided context. '
    'If the answer is not in the context, say that clearly. '
    'Do not present coding guidance as clinical advice.'
  )

  response = ollama.chat(
    model=chat_model,
    messages=[
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': f'Question: {query}\n\nContext:\n\n' + '\n\n'.join(context_lines)},
    ],
    stream=True,
  )

  print('\nAnswer:')
  for chunk in response:
    print(chunk['message']['content'], end='', flush=True)
  print()


def run_self_check():
  documents = [
    MedicalCodeDocument(
      code='E11.9',
      title='Type 2 diabetes mellitus without complications',
      description='Type 2 diabetes with no documented complication.',
      synonyms=['type 2 diabetes', 'adult-onset diabetes', 'high blood sugar'],
      source='self-check',
    ),
    MedicalCodeDocument(
      code='I10',
      title='Essential hypertension',
      description='Primary high blood pressure without a more specific secondary cause.',
      synonyms=['high blood pressure', 'hypertension'],
      source='self-check',
    ),
  ]

  indexed_documents, idf, semantic_enabled = build_index(documents, DEFAULT_EMBED_MODEL, enable_semantic=False)
  results, _ = retrieve(
    query='high blood sugar from type 2 diabetes',
    indexed_documents=indexed_documents,
    idf=idf,
    embed_model=DEFAULT_EMBED_MODEL,
    top_n=1,
    keyword_weight=1.0,
    semantic_weight=0.0,
    semantic_enabled=semantic_enabled,
  )

  assert results[0]['document'].code == 'E11.9'
  print('Self-check passed.')


def run_query(args):
  documents = load_documents(args.data_path)
  indexed_documents, idf, semantic_enabled = build_index(
    documents=documents,
    embed_model=args.embed_model,
    enable_semantic=not args.keyword_only,
  )

  print(f'Loaded {len(documents)} medical code records from {args.data_path}')
  if semantic_enabled:
    print(f'Hybrid mode: keyword + embeddings ({args.embed_model})')
  else:
    print('Keyword-only mode')

  if args.query:
    results, semantic_enabled = retrieve(
      query=args.query,
      indexed_documents=indexed_documents,
      idf=idf,
      embed_model=args.embed_model,
      top_n=args.top_n,
      keyword_weight=args.keyword_weight,
      semantic_weight=args.semantic_weight,
      semantic_enabled=semantic_enabled,
    )
    print_results(results)
    if not args.no_chat:
      answer_query(args.query, results, args.chat_model)
    return

  while True:
    try:
      query = input('\nAsk about a medical code (or type exit): ').strip()
    except EOFError:
      print('\nGoodbye!')
      return

    if query.lower() in {'exit', 'quit'}:
      print('Goodbye!')
      return

    if not query:
      continue

    results, semantic_enabled = retrieve(
      query=query,
      indexed_documents=indexed_documents,
      idf=idf,
      embed_model=args.embed_model,
      top_n=args.top_n,
      keyword_weight=args.keyword_weight,
      semantic_weight=args.semantic_weight,
      semantic_enabled=semantic_enabled,
    )
    print_results(results)
    if not args.no_chat:
      answer_query(query, results, args.chat_model)


def parse_args():
  parser = argparse.ArgumentParser(description='Hybrid RAG starter project for medical codes.')
  parser.add_argument('--data-path', type=Path, default=DEFAULT_DATA_PATH)
  parser.add_argument('--query')
  parser.add_argument('--top-n', type=int, default=5)
  parser.add_argument('--embed-model', default=DEFAULT_EMBED_MODEL)
  parser.add_argument('--chat-model', default=DEFAULT_CHAT_MODEL)
  parser.add_argument('--keyword-weight', type=float, default=0.45)
  parser.add_argument('--semantic-weight', type=float, default=0.55)
  parser.add_argument('--keyword-only', action='store_true')
  parser.add_argument('--no-chat', action='store_true')
  parser.add_argument('--self-check', action='store_true')
  return parser.parse_args()


def main():
  args = parse_args()

  if args.self_check:
    run_self_check()
    return

  if args.top_n < 1:
    raise ValueError('--top-n must be at least 1')

  if args.keyword_weight < 0 or args.semantic_weight < 0:
    raise ValueError('Weights must be non-negative')

  if args.keyword_weight == 0 and args.semantic_weight == 0:
    raise ValueError('At least one retrieval weight must be greater than 0')

  run_query(args)


if __name__ == '__main__':
  main()
