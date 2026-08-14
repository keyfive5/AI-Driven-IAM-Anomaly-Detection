/**
 * The normalised event schema and the identity-action knowledge base.
 *
 * Every supported log format (CloudTrail, Azure Activity, Entra ID sign-ins,
 * Okta System Log, GCP audit, CyberArk, generic CSV/JSONL) is flattened into a
 * single flat record so that detection logic never has to care where an event
 * came from.
 *
 * Normalised event
 * ----------------
 *   id           string    stable identifier
 *   ts           number    epoch milliseconds (UTC)
 *   actor        string    principal that performed the action
 *   actorType    string    Human | Role | Service | Root | Unknown
 *   action       string    "service:Operation", e.g. "iam:CreateAccessKey"
 *   service      string    "iam", "s3", "signin", "cyberark", ...
 *   resource     string    target object, best-effort
 *   ip           string    source address
 *   region       string    cloud region / datacentre
 *   country      string?   ISO-ish country label when the source provides it
 *   lat, lon     number?   coordinates when the source provides them
 *   userAgent    string
 *   outcome      "success" | "failure"
 *   errorCode    string?
 *   mfa          boolean?  null when the source does not say
 *   sessionId    string?
 *   source       string    originating log format
 *   label        0 | 1     ground truth when known (synthetic / annotated logs)
 *   campaign     string?   ground-truth campaign id
 *   stage        string?   ground-truth kill-chain stage
 *   raw          object    the untouched source record
 */

export const ACTOR_TYPES = ['Human', 'Role', 'Service', 'Root', 'Unknown'];

/** MITRE ATT&CK tactics, in kill-chain order. Used to sequence attack chains. */
export const TACTICS = [
  'Reconnaissance',
  'Initial Access',
  'Execution',
  'Persistence',
  'Privilege Escalation',
  'Defense Evasion',
  'Credential Access',
  'Discovery',
  'Lateral Movement',
  'Collection',
  'Exfiltration',
  'Impact',
];

export const TACTIC_ORDER = Object.fromEntries(TACTICS.map((t, i) => [t, i]));

/**
 * Sensitivity catalogue.
 *
 * `sensitivity` (0..1) expresses how much damage the operation can do if it is
 * the wrong person doing it. It is a domain prior, not a learned quantity — the
 * models supply "is this unusual", this table supplies "does unusual matter
 * here". A rare `s3:ListBucket` is noise; a rare `iam:AttachUserPolicy` is not.
 */
const SENSITIVE_OPS = {
  // Identity & privilege
  'iam:CreateUser': 0.85,
  'iam:DeleteUser': 0.8,
  'iam:CreateAccessKey': 0.95,
  'iam:UpdateAccessKey': 0.8,
  'iam:CreateLoginProfile': 0.9,
  'iam:UpdateLoginProfile': 0.9,
  'iam:AttachUserPolicy': 0.95,
  'iam:AttachRolePolicy': 0.95,
  'iam:AttachGroupPolicy': 0.9,
  'iam:PutUserPolicy': 0.95,
  'iam:PutRolePolicy': 0.95,
  'iam:CreateRole': 0.8,
  'iam:UpdateAssumeRolePolicy': 0.9,
  'iam:AddUserToGroup': 0.8,
  'iam:CreatePolicyVersion': 0.85,
  'iam:SetDefaultPolicyVersion': 0.9,
  'iam:DeactivateMFADevice': 0.95,
  'iam:DeleteVirtualMFADevice': 0.9,
  'iam:ChangePassword': 0.4,
  'sts:AssumeRole': 0.35,
  'sts:GetFederationToken': 0.6,
  // Detection & logging
  'cloudtrail:StopLogging': 1.0,
  'cloudtrail:DeleteTrail': 1.0,
  'cloudtrail:UpdateTrail': 0.85,
  'cloudtrail:PutEventSelectors': 0.8,
  'guardduty:DeleteDetector': 1.0,
  'guardduty:UpdateDetector': 0.85,
  'config:DeleteConfigurationRecorder': 0.9,
  'logs:DeleteLogGroup': 0.8,
  // Secrets
  'secretsmanager:GetSecretValue': 0.7,
  'secretsmanager:PutSecretValue': 0.6,
  'ssm:GetParameters': 0.6,
  'ssm:GetParameter': 0.55,
  'kms:Decrypt': 0.5,
  'kms:ScheduleKeyDeletion': 0.9,
  'kms:PutKeyPolicy': 0.85,
  // Data movement
  's3:PutBucketPolicy': 0.85,
  's3:PutBucketAcl': 0.85,
  's3:DeleteBucketPolicy': 0.85,
  's3:GetObject': 0.3,
  's3:PutObject': 0.35,
  's3:DeleteObject': 0.5,
  'ec2:CreateSnapshot': 0.55,
  'ec2:ModifySnapshotAttribute': 0.9,
  'ec2:ModifyImageAttribute': 0.85,
  'ec2:CreateKeyPair': 0.7,
  'ec2:AuthorizeSecurityGroupIngress': 0.7,
  'ec2:RunInstances': 0.5,
  'ec2:TerminateInstances': 0.6,
  'rds:CreateDBSnapshot': 0.55,
  'rds:ModifyDBSnapshotAttribute': 0.9,
  // Sign-in
  'signin:ConsoleLogin': 0.45,
  'signin:SignIn': 0.45,
  'signin:MfaChallenge': 0.4,
  // Directory (Okta / Entra canonical verbs)
  'okta:SignIn': 0.45,
  'okta:SignOut': 0.15,
  'okta:GrantPrivilege': 0.95,
  'okta:RevokePrivilege': 0.7,
  'okta:DeactivateMFADevice': 0.95,
  'okta:ResetMFADevice': 0.7,
  'okta:ResetPassword': 0.65,
  'okta:CreateUser': 0.8,
  'okta:DeleteUser': 0.7,
  'okta:CreateApiToken': 0.9,
  'okta:AddGroupMembership': 0.7,
  'okta:AddAppMembership': 0.5,
  'okta:DeletePolicy': 0.85,
  // Privileged access management
  'cyberark:RetrievePassword': 0.8,
  'cyberark:Connect': 0.6,
  'cyberark:AddSafeMember': 0.9,
  'cyberark:UpdateSafeMember': 0.85,
  'cyberark:DeleteSafeMember': 0.8,
  'cyberark:ChangePassword': 0.5,
  'cyberark:RotatePassword': 0.45,
  'cyberark:Logon': 0.35,
  'cyberark:ExportVault': 1.0,
};

/** Verb prefixes that mutate state. Used when an operation is not catalogued. */
const WRITE_PREFIXES = [
  'create', 'delete', 'put', 'update', 'modify', 'attach', 'detach', 'add',
  'remove', 'set', 'stop', 'start', 'terminate', 'deactivate', 'activate',
  'write', 'authorize', 'revoke', 'disable', 'enable', 'reset', 'rotate',
  'change', 'assign', 'grant', 'import', 'restore', 'associate', 'run',
];

const READ_PREFIXES = ['get', 'list', 'describe', 'read', 'head', 'lookup', 'search', 'view', 'query', 'batchget'];

/** Operations that read configuration — the raw material of reconnaissance. */
export function isDiscoveryAction(action) {
  const op = operationOf(action).toLowerCase();
  return op.startsWith('list') || op.startsWith('describe') || op.startsWith('get') || op.startsWith('search');
}

export function serviceOf(action) {
  const i = action.indexOf(':');
  return i === -1 ? 'unknown' : action.slice(0, i);
}

export function operationOf(action) {
  const i = action.indexOf(':');
  return i === -1 ? action : action.slice(i + 1);
}

export function isWriteAction(action) {
  const op = operationOf(action).toLowerCase();
  if (READ_PREFIXES.some((p) => op.startsWith(p))) return false;
  return WRITE_PREFIXES.some((p) => op.startsWith(p));
}

/**
 * Sensitivity of an operation in 0..1. Catalogued operations win; anything else
 * falls back to a service-level prior blended with read/write.
 */
export function sensitivityOf(action) {
  if (SENSITIVE_OPS[action] !== undefined) return SENSITIVE_OPS[action];
  const svc = serviceOf(action);
  const base = SERVICE_PRIOR[svc] ?? 0.25;
  return Math.min(1, isWriteAction(action) ? base + 0.2 : base);
}

const SERVICE_PRIOR = {
  iam: 0.6,
  sts: 0.4,
  organizations: 0.6,
  cloudtrail: 0.7,
  guardduty: 0.65,
  config: 0.5,
  kms: 0.5,
  secretsmanager: 0.55,
  ssm: 0.4,
  s3: 0.3,
  ec2: 0.35,
  rds: 0.35,
  lambda: 0.4,
  signin: 0.4,
  cyberark: 0.55,
  entra: 0.45,
  azure: 0.35,
  okta: 0.45,
  gcp: 0.35,
};

export const SEVERITY_LEVELS = ['info', 'low', 'medium', 'high', 'critical'];

export function severityFromRisk(risk) {
  if (risk >= 85) return 'critical';
  if (risk >= 70) return 'high';
  if (risk >= 50) return 'medium';
  if (risk >= 30) return 'low';
  return 'info';
}

/** RFC1918 / loopback / link-local / CGNAT. */
export function isPrivateIp(ip) {
  if (!ip || typeof ip !== 'string') return false;
  if (ip.includes(':')) return ip === '::1' || ip.startsWith('fd') || ip.startsWith('fe80');
  const p = ip.split('.').map(Number);
  if (p.length !== 4 || p.some((n) => Number.isNaN(n))) return false;
  if (p[0] === 10 || p[0] === 127) return true;
  if (p[0] === 192 && p[1] === 168) return true;
  if (p[0] === 172 && p[1] >= 16 && p[1] <= 31) return true;
  if (p[0] === 169 && p[1] === 254) return true;
  if (p[0] === 100 && p[1] >= 64 && p[1] <= 127) return true;
  return false;
}

/** Coarse client class from a user agent — console vs SDK vs script vs browser. */
export function clientClass(ua) {
  if (!ua) return 'unknown';
  const s = String(ua).toLowerCase();
  if (s.includes('console') || s.includes('signin.amazonaws') || s.includes('portal.azure')) return 'console';
  if (s.includes('aws-cli') || s.includes('az-cli') || s.includes('gcloud')) return 'cli';
  if (s.includes('boto') || s.includes('botocore') || s.includes('aws-sdk') || s.includes('python-requests') || s.includes('msgraph')) return 'sdk';
  if (s.includes('terraform') || s.includes('pulumi') || s.includes('cloudformation') || s.includes('ansible')) return 'iac';
  if (s.includes('curl') || s.includes('wget') || s.includes('powershell') || s.includes('go-http')) return 'script';
  if (s.includes('mozilla') || s.includes('chrome') || s.includes('safari') || s.includes('edg/')) return 'browser';
  if (s.includes('cyberark') || s.includes('psmp')) return 'pam';
  return 'other';
}

/** Great-circle distance in km, used by the impossible-travel detector. */
export function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}
