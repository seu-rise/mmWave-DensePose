"""Explicit content verification and readiness against the frozen release baseline."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time
import uuid

CHUNK_BYTES=8*1024*1024
RELEASE_DIRS={"raw","keypoint","keypoint_muti","dense"}

class VerificationError(ValueError):
    pass
class ScanCancelled(Exception):
    pass

def digest(path, cancel, report):
    h=hashlib.sha256(); size=0
    with path.open('rb') as stream:
        while True:
            if cancel.is_set():raise ScanCancelled
            block=stream.read(CHUNK_BYTES)
            if not block:break
            h.update(block);size+=len(block);report(len(block))
    if path.stat().st_size!=size:raise OSError('File changed while hashing')
    return h.hexdigest()

def relative(value):
    if not isinstance(value,str) or not value or '\\' in value or '\x00' in value:raise VerificationError('Invalid release path')
    p=PurePosixPath(value)
    if p.is_absolute() or any(part in ('','.', '..') for part in p.parts):raise VerificationError('Unsafe release path')
    return Path(*p.parts)

def load_json(path):
    try:value=json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:raise VerificationError('Release metadata unavailable') from exc
    if not isinstance(value,dict):raise VerificationError('Release metadata must be an object')
    return value

class DatasetVerificationService:
    """The release summary is static. Files are read only after ``start``."""
    def __init__(self,repository_root,dataset_root,report_root,*,trust_pickle=False):
        self.repository_root=Path(repository_root).resolve();self.dataset_root=Path(dataset_root).resolve();self.report_root=Path(report_root).resolve();self.trust_pickle=bool(trust_pickle)
        self.release_path=self.repository_root/'dataset'/'release-manifest.json';self.baseline_path=self.repository_root/'dataset'/'release-baseline.json'
        self.lock=threading.RLock();self.jobs={};self.threads={};self._load_recent()
    def _baseline(self):
        release=load_json(self.release_path);expected=release.get('dataset_manifest_sha256')
        if not isinstance(expected,str) or len(expected)!=64 or digest_static(self.baseline_path).lower()!=expected.lower():raise VerificationError('Frozen baseline fingerprint mismatch')
        manifest=load_json(self.baseline_path);raw=manifest.get('sequences');files=manifest.get('files')
        if manifest.get('dataset_kind')!='full_multimodal' or not isinstance(raw,dict) or not isinstance(files,dict):raise VerificationError('Frozen baseline is not a full multimodal release')
        inventory={}
        for name,item in files.items():
            rel=relative(name)
            if not isinstance(item,dict) or not isinstance(item.get('size_bytes'),int) or item['size_bytes']<0 or not isinstance(item.get('sha256'),str) or len(item['sha256'])!=64:raise VerificationError('Invalid frozen file inventory')
            inventory[rel.as_posix()]={'size':item['size_bytes'],'sha256':item['sha256'].lower()}
        sequence_bytes=0
        for item in raw.values():
            if not isinstance(item,dict):raise VerificationError('Invalid frozen sequence metadata')
            rel=relative(item.get('file'));size=item.get('size_bytes');sha=item.get('sha256')
            if not isinstance(size,int) or size<0 or not isinstance(sha,str) or len(sha)!=64:raise VerificationError('Invalid frozen sequence inventory')
            entry={'size':size,'sha256':sha.lower()};actual=inventory.get(rel.as_posix())
            if actual is not None and actual!=entry:raise VerificationError('Frozen sequence metadata differs from release inventory')
            inventory[rel.as_posix()]=entry
            sequence_bytes+=size
        if len(raw)!=release.get('expected_sequence_count') or sequence_bytes!=release.get('expected_sequence_bytes'):raise VerificationError('Frozen release sequence count or bytes mismatch')
        return release,manifest,inventory
    def summary(self):
        release, manifest, inventory = self._baseline()
        subjects = {
            str(sequence.get('subject_id'))
            for sequence in manifest['sequences'].values()
            if sequence.get('subject_id')
        }
        overview_path = self.repository_root / 'dataset' / 'release-overview.json'
        overview = load_json(overview_path) if overview_path.is_file() else {}
        overview_splits = overview.get('splits', [])
        splits = (overview_splits
                 if overview.get('baseline_sha256') == release['dataset_manifest_sha256']
                 and isinstance(overview_splits, list) else [])
        single_frame_split = next(
            (item for item in splits if isinstance(item, dict)
             and item.get('id') == 'single_intra_s42' and item.get('seq_len') == 1),
            None,
        )
        split_index = manifest.get('frozen_indexes', {}).get('single_intra_s42')
        pinned_index = inventory.get(split_index)
        split_index_matches = (pinned_index is not None and single_frame_split is not None
                               and single_frame_split.get('source_index_sha256') == pinned_index['sha256'])
        valid_counts = (single_frame_split or {}).get('counts', {})
        if not isinstance(valid_counts, dict):
            valid_counts = {}
        split_names = ('train', 'val', 'test')
        valid_frames = None
        if split_index_matches and all(
                type(valid_counts.get(name)) is int and valid_counts[name] >= 0
                for name in split_names):
            valid_frames = sum(valid_counts[name] for name in split_names)
        stored_frames = sum(int(sequence.get('frames', 0)) for sequence in manifest['sequences'].values())
        return {
            'dataset_id': release.get('dataset', 'mmWave-DensePose'),
            'release_version': release.get('dataset_version'),
            'release_status': release.get('status'),
            'baseline_sha256': release['dataset_manifest_sha256'],
            'generated_at': manifest.get('generated_at'),
            'statistics_source': 'frozen_release_baseline',
            'sequences': len(manifest['sequences']),
            'frames': stored_frames,
            'stored_frames': stored_frames,
            'valid_synchronized_frames': valid_frames,
            'valid_frames_source': 'single_intra_s42' if valid_frames is not None else None,
            'subjects': len(subjects),
            'sequence_bytes': release['expected_sequence_bytes'],
            'modalities': list(manifest.get('modalities', [])),
            'indexes': [name for name in manifest['files']
                        if name.startswith(('keypoint/', 'keypoint_muti/', 'dense/'))],
            'splits': splits,
        }
    def _public(self,job):return {k:v for k,v in job.items() if not k.startswith('_')}
    def start(self):
        with self.lock:
            if any(j['status']=='scanning' for j in self.jobs.values()):raise VerificationError('Dataset verification is already running')
            jid=uuid.uuid4().hex;cancel=threading.Event();job={'id':jid,'status':'scanning','stage':'release baseline','checked':0,'total':None,'bytes_hashed':0,'findings':[],'started_at':time.time(),'finished_at':None,'report':None,'_cancel':cancel};self.jobs[jid]=job
            thread=threading.Thread(target=self._scan,args=(jid,cancel),name=f'dataset-verify-{jid[:8]}',daemon=True);self.threads[jid]=thread;thread.start();return self._public(job)
    def status(self,jid):
        with self.lock:return self._public(self.jobs[jid]) if jid in self.jobs else None
    def latest(self):
        with self.lock:
            values=sorted(self.jobs.values(),key=lambda item:item.get('started_at',0),reverse=True)
            return self._public(values[0]) if values else None
    def cancel(self,jid):
        with self.lock:
            job=self.jobs.get(jid)
            if job is None:return None
            if job['status']=='scanning':job['_cancel'].set()
            return self._public(job)
    def close(self):
        with self.lock:
            jobs=list(self.jobs.values())
            for job in jobs:
                if job['status']=='scanning':job['_cancel'].set()
            threads=list(self.threads.values())
        for thread in threads:thread.join(timeout=2)
    def _safe_target(self,relative_name):
        rel=relative(relative_name);root=self.dataset_root
        if root.is_symlink():raise OSError('Configured root is a symbolic link')
        current=root
        for part in rel.parts:
            current=current/part
            if current.is_symlink():raise OSError('Symbolic link in release path')
        resolved=current.resolve(strict=False)
        if resolved!=root and root not in resolved.parents:raise OSError('Release path escaped configured root')
        return resolved
    def _actual_files(self,cancel):
        if not self.dataset_root.is_dir() or self.dataset_root.is_symlink():raise OSError('Dataset root is not a regular directory')
        found=set();issues=[];todo=[(self.dataset_root,'')]
        while todo:
            if cancel.is_set():raise ScanCancelled
            directory,prefix=todo.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    rel=f'{prefix}/{entry.name}'.lstrip('/')
                    if entry.is_symlink():issues.append({'path':rel,'severity':'error','expected':'Regular release entry','actual':'Symlink','cause':'Symbolic links are not part of the release inventory.','resolution':'Replace it with a regular file from the selected release.'});continue
                    if not prefix and (entry.name in {'cache','completion_cache','preparation-progress.json'}):continue
                    if not prefix and entry.is_dir(follow_symlinks=False) and entry.name not in RELEASE_DIRS:
                        issues.append({'path':rel+'/','severity':'warning','expected':'No extra release directory','actual':'Additional directory','cause':'This directory is outside the frozen release layout.','resolution':'Review it; derived caches are excluded from verification.'})
                        continue
                    if entry.is_dir(follow_symlinks=False):todo.append((Path(entry.path),rel))
                    elif entry.is_file(follow_symlinks=False):found.add(rel)
                    else:issues.append({'path':rel,'severity':'warning','expected':'Regular file','actual':'Unsupported entry','cause':'This entry is not a regular release file.','resolution':'Review it; do not remove it automatically.'})
        return found,issues
    def _write(self,job):
        folder=self.report_root/job['id'];folder.mkdir(parents=True,exist_ok=True);temp=folder/'report.json.tmp';temp.write_text(json.dumps(self._public(job),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');temp.replace(folder/'report.json')
    def _scan(self,jid,cancel):
        job=self.jobs[jid]
        terminal_status=None
        try:
            release,reference,inventory=self._baseline();job.update(stage='enumerating files',release_version=release['dataset_version'],baseline_sha256=release['dataset_manifest_sha256'],target_root_label=self.dataset_root.name,expected_sequence_count=len(reference['sequences']))
            actual,findings=self._actual_files(cancel);required=set(inventory)|{'manifest.json'};job.update(stage='checking file hashes',total=len(required),findings=findings)
            for rel in sorted(required & actual):
                if cancel.is_set():raise ScanCancelled
                path=self._safe_target(rel)
                expected={'size':self.baseline_path.stat().st_size,'sha256':hashlib.sha256(self.baseline_path.read_bytes()).hexdigest()} if rel=='manifest.json' else inventory[rel]
                try:
                    size=path.stat().st_size
                    if size!=expected['size']:
                        findings.append({'path':rel,'severity':'error','expected':{'size_bytes':expected['size']},'actual':{'size_bytes':size},'cause':'File size differs from the selected release.','resolution':'Restore this file from the selected release, then verify again.'})
                    else:
                        before=path.stat()
                        actual_sha=digest(path,cancel,lambda n:job.__setitem__('bytes_hashed',job['bytes_hashed']+n))
                        after=path.stat()
                        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                            findings.append({'path':rel,'severity':'unverified','expected':expected,'actual':'Changed during scan','cause':'The file changed while its checksum was being read.','resolution':'Wait for copying or writing to finish, then verify again.'})
                        elif actual_sha!=expected['sha256']:
                            findings.append({'path':rel,'severity':'error','expected':{'sha256':expected['sha256']},'actual':{'sha256':actual_sha},'cause':'File contents differ from the frozen release.','resolution':'Restore the matching release file, then verify again.'})
                except ScanCancelled:raise
                except OSError as exc:findings.append({'path':rel,'severity':'unverified','expected':expected,'actual':exc.strerror or exc.__class__.__name__,'cause':'The file could not be read for its complete hash.','resolution':'Wait for writes to finish, grant read access and rerun.'})
                job['checked']+=1
            for rel in sorted(required-actual):
                findings.append({'path':rel,'severity':'error','expected':inventory.get(rel,{'size':self.baseline_path.stat().st_size}),'actual':'Missing','cause':'Required release file is missing.','resolution':'Restore this path from the selected release.'})
                job['checked']+=1
            for rel in sorted(actual-required):findings.append({'path':rel,'severity':'warning','expected':'Not in release inventory','actual':'Additional file','cause':'The selected root contains an additional file.','resolution':'Review it; caches and preparation progress are excluded.'})
            manifest_problem=any(f.get('path')=='manifest.json' and f.get('severity') in ('error','unverified') for f in findings)
            if not manifest_problem and 'manifest.json' in actual:
                # The manifest bytes have already matched the independently pinned SHA256;
                # its parsed metadata is therefore exactly the frozen baseline.
                if reference.get('dataset_kind')!='full_multimodal':
                    findings.append({'path':'manifest.json','severity':'error','expected':'full_multimodal','actual':reference.get('dataset_kind'),'cause':'The release manifest is not a full multimodal dataset.','resolution':'Restore the matching release manifest.'})
            errors=sum(f['severity']=='error' for f in findings);warnings=sum(f['severity']=='warning' for f in findings);unverified=sum(f['severity']=='unverified' for f in findings)
            terminal_status='consistent' if not(errors or unverified) else 'mismatch'
            job.update(stage='saving verification',findings=findings,errors=errors,warnings=warnings,unverified=unverified)
        except ScanCancelled:job.update(stage='cancelled',status='cancelled')
        except Exception as exc:job.update(stage='unable to confirm',status='cannot_confirm',failure=exc.strerror if isinstance(exc,OSError) and exc.strerror else exc.__class__.__name__)
        finally:
            with self.lock:
                if terminal_status is not None:job.update(stage='complete',status=terminal_status)
                job['finished_at']=time.time()
                try:
                    self._write(job)
                    if job.get('status')=='consistent' and not cancel.is_set():
                        # A standalone Verify can finish a previously interrupted
                        # installation without downloading its already valid files.
                        from mmwave_densepose.data.download import read_installation_state, write_installation_state
                        try:receipt=read_installation_state(self.dataset_root,job['baseline_sha256'])
                        except ValueError:receipt=None
                        if receipt is not None and receipt.get('complete') is False:
                            write_installation_state(self.dataset_root,job['baseline_sha256'],
                                                     installed_archive_ids=receipt['installed_archive_ids'],complete=True)
                except OSError as exc:
                    job.update(stage='verification state unavailable',status='failed',failure=exc.strerror or exc.__class__.__name__)
    def _load_recent(self):
        if not self.report_root.is_dir() or self.report_root.is_symlink():return
        for path in sorted(self.report_root.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True):
            if len(path.name)!=32 or path.is_symlink():continue
            report=path/'report.json'
            if report.is_file() and not report.is_symlink():
                try:
                    job=load_json(report)
                    if job.get('id')==path.name and job.get('status')!='scanning':job['_cancel']=threading.Event();self.jobs[path.name]=job
                except VerificationError:continue

def digest_static(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(CHUNK_BYTES),b''):h.update(chunk)
    return h.hexdigest()
