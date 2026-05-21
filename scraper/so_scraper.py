#!/usr/bin/env python3
"""
StackOverflow Question Scraper

Crawls StackOverflow questions with accepted answers for use as seed data
in the Terminal-Lego task generation pipeline.

Usage:
    python so_scraper.py --round 1 --output ./data --count 5000
"""

import json
import requests
import time
import random
import re
import sys
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any


SO_API_BASE = "https://api.stackexchange.com/2.3"

TAGS_BY_PRIORITY = [
    ('linux', 800), ('bash', 800), ('python', 800), ('git', 600), ('docker', 400),
    ('ssh', 300), ('nginx', 300), ('ubuntu', 300), ('debian', 200), ('centos', 200),
    ('systemd', 200), ('cron', 200), ('chmod', 150), ('sudo', 150),
    ('networking', 300), ('tcp', 200), ('http', 300), ('https', 200), ('dns', 200),
    ('ssl', 250), ('ssl-certificate', 200), ('openssl', 250), ('curl', 250), ('wget', 150),
    ('file-io', 300), ('tar', 150), ('gzip', 100), ('zip', 150),
    ('pandas', 400), ('numpy', 350), ('matplotlib', 250), ('jupyter-notebook', 250),
    ('pip', 250), ('virtualenv', 200), ('conda', 200),
    ('pytorch', 350), ('tensorflow', 300), ('scikit-learn', 300),
    ('machine-learning', 400), ('deep-learning', 300), ('keras', 250),
    ('json', 300), ('csv', 250), ('regex', 350), ('xml', 200),
    ('encryption', 200), ('cryptography', 200), ('hash', 150), ('security', 250),
    ('debugging', 300), ('logging', 200), ('exception', 200),
    ('sqlite', 250), ('mysql', 300), ('postgresql', 300), ('redis', 200), ('mongodb', 200),
    ('web-scraping', 250), ('requests', 200), ('beautifulsoup', 200),
    ('selenium', 200), ('api', 250),
    ('algorithm', 350), ('sorting', 200), ('recursion', 200), ('data-structures', 250),
    ('multiprocessing', 200), ('multithreading', 200), ('asyncio', 200),
    ('string', 300), ('sed', 200), ('awk', 200), ('grep', 200),
    ('ffmpeg', 200), ('opencv', 250), ('image-processing', 200),
    ('makefile', 200), ('cmake', 200), ('gcc', 200),
    ('vim', 200), ('tmux', 150),
    ('r', 250), ('scipy', 200), ('sympy', 150),
    ('shell', 400), ('command-line', 300), ('terminal', 200), ('unix', 300),
    ('iptables', 150), ('firewall', 150), ('apache', 250),
    ('qemu', 100), ('virtualization', 150), ('kubernetes', 200),
    ('yaml', 200), ('nlp', 200), ('huggingface-transformers', 150), ('parquet', 100),
]


def get_questions_by_tag(
    tag: str,
    api_key: str,
    min_score: int = 10,
    page_size: int = 100,
    max_pages: int = 20,
    sort: str = 'activity',
    start_page: int = 1,
    exclude_ids: set = None,
) -> tuple:
    questions = []
    should_stop = False
    if exclude_ids is None:
        exclude_ids = set()

    for page in range(start_page, start_page + max_pages):
        params = {
            'order': 'desc',
            'sort': sort,
            'tagged': tag,
            'site': 'stackoverflow',
            'filter': 'withbody',
            'pagesize': page_size,
            'page': page,
            'min': min_score,
            'key': api_key,
        }

        try:
            response = requests.get(
                f"{SO_API_BASE}/questions",
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                print(f"  Rate limited, waiting 60s...")
                time.sleep(60)
                continue

            if response.status_code == 400:
                print(f"  Tag '{tag}' invalid, skipping")
                return questions, False

            response.raise_for_status()
            data = response.json()

            if 'backoff' in data:
                print(f"  Backoff {data['backoff']}s...")
                time.sleep(data['backoff'])

            if 'items' not in data:
                break

            for item in data['items']:
                if not item.get('accepted_answer_id'):
                    continue
                if item['question_id'] in exclude_ids:
                    continue

                questions.append({
                    'question_id': item['question_id'],
                    'title': item['title'],
                    'link': item['link'],
                    'score': item['score'],
                    'view_count': item.get('view_count', 0),
                    'answer_count': item.get('answer_count', 0),
                    'accepted_answer_id': item.get('accepted_answer_id'),
                    'tags': item.get('tags', []),
                    'body': item.get('body', ''),
                    'creation_date': item.get('creation_date'),
                })

            quota = data.get('quota_remaining', 0)
            print(f"    page {page}: +{len(data['items'])} items, quota={quota}")

            if quota < 100:
                print("  Quota low, stopping")
                return questions, True

            if not data.get('has_more', False):
                break

        except Exception as e:
            print(f"  Error: {e}")
            break

        time.sleep(0.5)

    return questions, should_stop


def get_answers_batch(answer_ids: List[int], api_key: str) -> Dict[int, Dict]:
    """Fetch answers in batch (max 30 per request)"""
    results = {}
    ids_str = ';'.join(str(aid) for aid in answer_ids)
    params = {
        'site': 'stackoverflow',
        'filter': 'withbody',
        'key': api_key,
    }

    try:
        response = requests.get(
            f"{SO_API_BASE}/answers/{ids_str}",
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if 'backoff' in data:
                time.sleep(data['backoff'])
            for item in data.get('items', []):
                results[item['answer_id']] = {
                    'answer_id': item['answer_id'],
                    'body': item.get('body', ''),
                    'score': item.get('score', 0),
                }
    except Exception as e:
        print(f"  Failed to fetch answers batch: {e}")

    return results


def clean_html(html_text: str) -> str:
    clean = re.sub(r'<[^>]+>', '', html_text)
    clean = clean.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    clean = clean.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    return clean.strip()


def save_data(questions, output_path, jsonl_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total': len(questions),
                'with_answers': sum(1 for q in questions if q.get('accepted_answer')),
                'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'questions': questions
        }, f, ensure_ascii=False, indent=2)

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for q in questions:
            item = {
                'question_id': q['question_id'],
                'title': q['title'],
                'link': q['link'],
                'score': q['score'],
                'tags': q['tags'],
                'search_tag': q.get('search_tag', ''),
                'body_text': clean_html(q.get('body', ''))[:1000],
                'answer_text': clean_html(q.get('accepted_answer', {}).get('body', ''))[:2000] if q.get('accepted_answer') else '',
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(questions)} questions to {output_path}")


def load_existing_ids(dedup_dir: Path) -> set:
    """Load question IDs from all JSON files in dedup_dir for deduplication"""
    exclude_ids = set()
    if not dedup_dir or not dedup_dir.exists():
        return exclude_ids

    for f in sorted(dedup_dir.glob("*.json")):
        try:
            with open(f, 'r') as fh:
                old_data = json.load(fh)
            old_qs = old_data.get('questions', [])
            exclude_ids.update(q['question_id'] for q in old_qs)
            print(f"Dedup loaded: {f.name} ({len(old_qs)} questions)")
        except Exception as e:
            print(f"Skip: {f}: {e}")

    return exclude_ids


def main():
    parser = argparse.ArgumentParser(description='StackOverflow Question Scraper')
    parser.add_argument('--round', '-r', type=int, default=1, help='Round number')
    parser.add_argument('--output', '-o', type=str, default='./data', help='Output directory')
    parser.add_argument('--count', '-c', type=int, default=5000, help='Target question count')
    parser.add_argument('--dedup-dir', type=str, default=None, help='Directory with existing JSON files for deduplication')
    parser.add_argument('--api-key', type=str, default=None, help='StackOverflow API key')

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('SO_API_KEY', '')
    if not api_key:
        print("Warning: No SO API key provided. Rate limits will be strict.")
        print("Get one at https://stackapps.com/ and pass via --api-key or SO_API_KEY env var.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"so_data_r{args.round}.json"
    output_jsonl = output_dir / f"so_data_r{args.round}.jsonl"

    tags = list(TAGS_BY_PRIORITY)
    random.shuffle(tags)

    dedup_dir = Path(args.dedup_dir) if args.dedup_dir else output_dir
    exclude_ids = load_existing_ids(dedup_dir)

    all_questions = []
    seen_ids = set(exclude_ids)
    target_count = args.count

    print("=" * 60)
    print(f"Round {args.round}: Crawling {target_count} StackOverflow questions")
    print(f"Excluding {len(exclude_ids)} existing IDs")
    print(f"Tags: {len(tags)} (shuffled)")
    print(f"Output: {output_json}")
    print("=" * 60)

    total_weight = sum(w for _, w in tags)

    for i, (tag, weight) in enumerate(tags, 1):
        if len(all_questions) >= target_count:
            print(f"\nReached target {target_count}!")
            break

        remaining = target_count - len(all_questions)
        tag_target = min(int(weight * target_count / total_weight * 2.0), remaining)

        print(f"\n[{i}/{len(tags)}] Tag: {tag} (target: ~{tag_target}, total: {len(all_questions)})")

        sort_method = random.choice(['activity', 'votes', 'creation'])
        start_page = random.randint(1, 5)

        if tag_target > 500:
            min_score, max_pages = 15, 25
        elif tag_target > 200:
            min_score, max_pages = 20, 15
        else:
            min_score, max_pages = 30, 10

        questions, should_stop = get_questions_by_tag(
            tag=tag,
            api_key=api_key,
            min_score=min_score,
            max_pages=max_pages,
            sort=sort_method,
            start_page=start_page,
            exclude_ids=seen_ids,
        )

        new_count = 0
        for q in questions:
            if q['question_id'] not in seen_ids:
                seen_ids.add(q['question_id'])
                q['search_tag'] = tag
                all_questions.append(q)
                new_count += 1
                if len(all_questions) >= target_count:
                    break

        print(f"  Found {len(questions)}, added {new_count} (sort={sort_method}, page={start_page})")

        if len(all_questions) % 1000 == 0 and len(all_questions) > 0:
            save_data(all_questions, str(output_json), str(output_jsonl))

        if should_stop:
            print("\nQuota low, saving and exiting")
            break

        time.sleep(1)

    # Batch fetch accepted answers
    print(f"\nFetching accepted answers for {len(all_questions)} questions...")

    need_answers = [(i, q['accepted_answer_id']) for i, q in enumerate(all_questions)
                    if q.get('accepted_answer_id') and not q.get('accepted_answer')]

    batch_size = 30
    fetched = 0
    for batch_start in range(0, len(need_answers), batch_size):
        batch = need_answers[batch_start:batch_start + batch_size]
        answer_ids = [aid for _, aid in batch]

        results = get_answers_batch(answer_ids, api_key)

        for idx, aid in batch:
            if aid in results:
                all_questions[idx]['accepted_answer'] = results[aid]
                fetched += 1

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Processed {batch_start + len(batch)}/{len(need_answers)}, fetched {fetched}")

        if (batch_start // batch_size) % 50 == 0 and batch_start > 0:
            save_data(all_questions, str(output_json), str(output_jsonl))

        time.sleep(0.5)

    save_data(all_questions, str(output_json), str(output_jsonl))

    with_answers = sum(1 for q in all_questions if q.get('accepted_answer'))
    print(f"\nDone! Crawled {len(all_questions)} questions ({with_answers} with answers)")
    print(f"Saved to: {output_json}")


if __name__ == "__main__":
    main()
