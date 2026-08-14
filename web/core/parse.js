/**
 * Log ingestion: sniff the format, then flatten to the normalised schema.
 *
 * Supported inputs
 *   - AWS CloudTrail            { "Records": [...] } or a bare array of records
 *   - Azure Activity Log        { "records": [...] } or a bare array
 *   - Microsoft Entra ID        sign-in logs (Graph export shape)
 *   - Okta System Log           array of events with eventType/actor/client
 *   - GCP Cloud Audit Log       protoPayload shape
 *   - CyberArk                  vault/PSM activity records (JSON or CSV)
 *   - Generic                   JSONL, JSON array, or CSV with recognisable columns
 *
 * Every parser is defensive: a malformed record is skipped and counted rather
 * than aborting the load, because real log exports always contain at least one
 * record that someone truncated.
 */

import { isPrivateIp, serviceOf } from './schema.js';

/** Country centroids for the locations the bundled corpora use. */
export const GEO = {
  'CA-ON': { lat: 43.65, lon: -79.38, country: 'Canada', city: 'Toronto' },
  'CA-BC': { lat: 49.28, lon: -123.12, country: 'Canada', city: 'Vancouver' },
  'US-VA': { lat: 38.83, lon: -77.31, country: 'United States', city: 'Ashburn' },
  'US-CA': { lat: 37.77, lon: -122.42, country: 'United States', city: 'San Francisco' },
  'US-NY': { lat: 40.71, lon: -74.01, country: 'United States', city: 'New York' },
  'IE-D': { lat: 53.35, lon: -6.26, country: 'Ireland', city: 'Dublin' },
  'DE-HE': { lat: 50.11, lon: 8.68, country: 'Germany', city: 'Frankfurt' },
  'GB-LND': { lat: 51.51, lon: -0.13, country: 'United Kingdom', city: 'London' },
  'SG': { lat: 1.35, lon: 103.82, country: 'Singapore', city: 'Singapore' },
  'IN-MH': { lat: 19.08, lon: 72.88, country: 'India', city: 'Mumbai' },
  'BR-SP': { lat: -23.55, lon: -46.63, country: 'Brazil', city: 'Sao Paulo' },
  'RU-MOW': { lat: 55.76, lon: 37.62, country: 'Russia', city: 'Moscow' },
  'NG-LA': { lat: 6.52, lon: 3.38, country: 'Nigeria', city: 'Lagos' },
  'CN-BJ': { lat: 39.9, lon: 116.4, country: 'China', city: 'Beijing' },
  'NL-NH': { lat: 52.37, lon: 4.89, country: 'Netherlands', city: 'Amsterdam' },
};

/** AWS region -> approximate location, so CloudTrail gets coarse geography. */
const REGION_GEO = {
  'us-east-1': 'US-VA',
  'us-east-2': 'US-VA',
  'us-west-1': 'US-CA',
  'us-west-2': 'US-CA',
  'ca-central-1': 'CA-ON',
  'eu-west-1': 'IE-D',
  'eu-west-2': 'GB-LND',
  'eu-central-1': 'DE-HE',
  'ap-southeast-1': 'SG',
  'ap-south-1': 'IN-MH',
  'sa-east-1': 'BR-SP',
};

let seq = 0;
const nextId = () => `e${(++seq).toString(36)}`;

export function resetIds() {
  seq = 0;
}

function toEpoch(value) {
  if (value === null || value === undefined) return NaN;
  if (typeof value === 'number') return value < 1e12 ? value * 1000 : value;
  const t = Date.parse(value);
  return Number.isNaN(t) ? NaN : t;
}

function str(value, fallback = '') {
  if (value === null || value === undefined) return fallback;
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return fallback;
}

/** Build a normalised event, filling in everything derivable. */
export function makeEvent(e) {
  const action = e.action || 'unknown:Unknown';
  const geoKey = e.geoKey || REGION_GEO[e.region];
  const geo = geoKey ? GEO[geoKey] : null;
  return {
    id: e.id || nextId(),
    ts: e.ts,
    actor: e.actor || 'unknown',
    actorType: e.actorType || 'Unknown',
    action,
    service: e.service || serviceOf(action),
    resource: e.resource || '-',
    ip: e.ip || '0.0.0.0',
    region: e.region || '-',
    country: e.country || geo?.country || null,
    city: e.city || geo?.city || null,
    lat: e.lat ?? geo?.lat ?? null,
    lon: e.lon ?? geo?.lon ?? null,
    userAgent: e.userAgent || '-',
    outcome: e.outcome === 'failure' ? 'failure' : 'success',
    errorCode: e.errorCode || null,
    mfa: e.mfa === undefined ? null : e.mfa,
    sessionId: e.sessionId || null,
    source: e.source || 'generic',
    label: e.label ? 1 : 0,
    campaign: e.campaign || null,
    stage: e.stage || null,
    private: isPrivateIp(e.ip),
    raw: e.raw || null,
  };
}

/* ------------------------------------------------------------------ AWS --- */

export function parseCloudTrail(record) {
  const ts = toEpoch(record.eventTime);
  if (Number.isNaN(ts)) return null;

  const ui = record.userIdentity || {};
  const sess = ui.sessionContext || {};
  const actor =
    ui.userName ||
    sess.sessionIssuer?.userName ||
    (ui.arn ? ui.arn.split('/').pop() : null) ||
    ui.principalId ||
    ui.accountId ||
    'unknown';

  let actorType = 'Human';
  if (ui.type === 'Root') actorType = 'Root';
  else if (ui.type === 'AssumedRole') actorType = 'Role';
  else if (ui.type === 'AWSService' || ui.invokedBy) actorType = 'Service';
  else if (ui.type === 'IAMUser') actorType = 'Human';
  else actorType = 'Unknown';

  const service = str(record.eventSource, '').replace(/\.amazonaws\.com$/, '') || 'aws';
  const op = str(record.eventName, 'Unknown');
  const action = `${service === 'signin' ? 'signin' : service}:${op}`;

  const resource =
    record.resources?.[0]?.ARN ||
    record.resources?.[0]?.resourceName ||
    record.requestParameters?.userName ||
    record.requestParameters?.roleName ||
    record.requestParameters?.bucketName ||
    record.requestParameters?.policyArn ||
    record.requestParameters?.instanceId ||
    '-';

  const errorCode = record.errorCode || (record.errorMessage ? 'Error' : null);
  const mfaRaw = sess.attributes?.mfaAuthenticated ?? record.additionalEventData?.MFAUsed;
  let mfa = null;
  if (mfaRaw !== undefined && mfaRaw !== null) {
    mfa = mfaRaw === true || mfaRaw === 'true' || mfaRaw === 'Yes';
  }

  return makeEvent({
    ts,
    actor: str(actor, 'unknown'),
    actorType,
    action,
    service: service === 'signin' ? 'signin' : service,
    resource: str(resource, '-'),
    ip: str(record.sourceIPAddress, '0.0.0.0'),
    region: str(record.awsRegion, '-'),
    userAgent: str(record.userAgent, '-'),
    outcome: errorCode ? 'failure' : 'success',
    errorCode,
    mfa,
    sessionId: str(record.requestID) || null,
    source: 'aws',
    id: record.eventID || undefined,
    raw: record,
  });
}

/* ---------------------------------------------------------------- Azure --- */

export function parseAzureActivity(record) {
  const ts = toEpoch(record.time || record.eventTimestamp);
  if (Number.isNaN(ts)) return null;

  const claims = record.identity?.claims || {};
  const actor =
    record.caller ||
    claims.upn ||
    claims.name ||
    claims['http://schemas.xmlsoap.org/ws/2005/05/identity/claims/upn'] ||
    record.identity?.principalId ||
    'unknown';

  const opName = str(record.operationName?.value || record.operationName, 'Unknown');
  const parts = opName.split('/');
  const provider = parts[0]?.replace(/^Microsoft\./, '').toLowerCase() || 'azure';
  const op = parts.slice(1).join('/') || opName;
  const resourceId = str(record.resourceId, '');
  const resultType = str(record.resultType || record.status?.value, 'Success');

  return makeEvent({
    ts,
    actor: str(actor, 'unknown'),
    actorType: String(actor).includes('@') ? 'Human' : 'Service',
    action: `azure-${provider}:${op}`,
    service: `azure-${provider}`,
    resource: resourceId ? resourceId.split('/').pop() : '-',
    ip: str(record.callerIpAddress, '0.0.0.0'),
    region: str(record.location, '-'),
    userAgent: str(record.properties?.userAgent, '-'),
    outcome: /succe/i.test(resultType) ? 'success' : 'failure',
    errorCode: /succe/i.test(resultType) ? null : resultType,
    sessionId: str(record.correlationId) || null,
    source: 'azure',
    raw: record,
  });
}

/** Microsoft Entra ID (Azure AD) sign-in logs. */
export function parseEntraSignIn(record) {
  const ts = toEpoch(record.createdDateTime || record.time);
  if (Number.isNaN(ts)) return null;

  const status = record.status || {};
  const failed = (status.errorCode ?? 0) !== 0;
  const loc = record.location || {};
  const authDetail = record.authenticationDetails?.[0];
  // `authenticationRequirement` is authoritative when present. Falling through
  // to the method list would turn an explicit "single factor" into "unknown",
  // which silently disarms every MFA-related detection on Entra logs.
  let mfa = null;
  if (record.authenticationRequirement === 'multiFactorAuthentication') mfa = true;
  else if (record.authenticationRequirement === 'singleFactorAuthentication') mfa = false;
  else if (authDetail?.authenticationMethod) mfa = !/password/i.test(authDetail.authenticationMethod);

  return makeEvent({
    ts,
    actor: str(record.userPrincipalName || record.userDisplayName, 'unknown'),
    actorType: 'Human',
    action: 'signin:SignIn',
    service: 'signin',
    resource: str(record.appDisplayName || record.resourceDisplayName, '-'),
    ip: str(record.ipAddress, '0.0.0.0'),
    region: str(loc.city || loc.state, '-'),
    country: str(loc.countryOrRegion) || null,
    city: str(loc.city) || null,
    lat: loc.geoCoordinates?.latitude ?? null,
    lon: loc.geoCoordinates?.longitude ?? null,
    userAgent: str(record.userAgent || record.deviceDetail?.browser, '-'),
    outcome: failed ? 'failure' : 'success',
    errorCode: failed ? String(status.errorCode) : null,
    mfa,
    sessionId: str(record.correlationId) || null,
    source: 'entra',
    raw: record,
  });
}

/* ----------------------------------------------------------------- Okta --- */

/**
 * Canonical operation names.
 *
 * Every directory calls the same act something different — AWS says
 * `iam:DeactivateMFADevice`, Okta says `user.mfa.factor.deactivate`, Entra says
 * something else again. Mapping them onto one vocabulary at ingest is what lets
 * a single rule pack cover every source instead of one pack per vendor.
 */
const OKTA_ACTIONS = [
  [/^user\.session\.start/, 'SignIn'],
  [/^user\.session\.end/, 'SignOut'],
  [/^user\.authentication/, 'SignIn'],
  [/^user\.mfa\.factor\.deactivate/, 'DeactivateMFADevice'],
  [/^user\.mfa\.factor\.(activate|reset)/, 'ResetMFADevice'],
  [/^user\.account\.privilege\.grant/, 'GrantPrivilege'],
  [/^user\.account\.privilege\.revoke/, 'RevokePrivilege'],
  [/^user\.account\.reset_password/, 'ResetPassword'],
  [/^user\.lifecycle\.create/, 'CreateUser'],
  [/^user\.lifecycle\.(delete|deactivate|suspend)/, 'DeleteUser'],
  [/^application\.user_membership\.add/, 'AddAppMembership'],
  [/^group\.user_membership\.add/, 'AddGroupMembership'],
  [/^system\.api_token\.create/, 'CreateApiToken'],
  [/^policy\.(rule\.)?(delete|deactivate)/, 'DeletePolicy'],
];

function canonicalOkta(eventType) {
  const type = str(eventType, 'event');
  for (const [rx, name] of OKTA_ACTIONS) if (rx.test(type)) return name;
  // Fall back to the last two segments in CamelCase: user.foo.bar → FooBar
  return type.split('.').slice(-2)
    .map((part) => part.replace(/(^|_)([a-z])/g, (_, __, c) => c.toUpperCase()))
    .join('') || 'Event';
}

export function parseOkta(record) {
  const ts = toEpoch(record.published);
  if (Number.isNaN(ts)) return null;

  const actor = record.actor || {};
  const client = record.client || {};
  const geo = client.geographicalContext || {};
  const outcome = record.outcome || {};
  const target = record.target?.[0];

  return makeEvent({
    ts,
    actor: str(actor.alternateId || actor.displayName || actor.id, 'unknown'),
    actorType: actor.type === 'User' ? 'Human' : 'Service',
    action: `okta:${canonicalOkta(record.eventType)}`,
    service: 'okta',
    resource: str(target?.alternateId || target?.displayName, '-'),
    ip: str(client.ipAddress, '0.0.0.0'),
    region: str(geo.state || geo.city, '-'),
    country: str(geo.country) || null,
    city: str(geo.city) || null,
    lat: geo.geolocation?.lat ?? null,
    lon: geo.geolocation?.lon ?? null,
    userAgent: str(client.userAgent?.rawUserAgent, '-'),
    outcome: /success/i.test(str(outcome.result, 'SUCCESS')) ? 'success' : 'failure',
    errorCode: /success/i.test(str(outcome.result, 'SUCCESS')) ? null : str(outcome.reason) || str(outcome.result),
    sessionId: str(record.authenticationContext?.externalSessionId) || null,
    source: 'okta',
    raw: record,
  });
}

/* ------------------------------------------------------------------ GCP --- */

export function parseGcpAudit(record) {
  const ts = toEpoch(record.timestamp || record.receiveTimestamp);
  if (Number.isNaN(ts)) return null;
  const p = record.protoPayload || {};
  const svc = str(p.serviceName, 'gcp').split('.')[0];
  const method = str(p.methodName, 'unknown').split('.').pop();
  const err = p.status?.code;

  return makeEvent({
    ts,
    actor: str(p.authenticationInfo?.principalEmail, 'unknown'),
    actorType: str(p.authenticationInfo?.principalEmail, '').includes('gserviceaccount') ? 'Service' : 'Human',
    action: `gcp-${svc}:${method}`,
    service: `gcp-${svc}`,
    resource: str(p.resourceName, '-').split('/').pop(),
    ip: str(p.requestMetadata?.callerIp, '0.0.0.0'),
    region: str(record.resource?.labels?.location, '-'),
    userAgent: str(p.requestMetadata?.callerSuppliedUserAgent, '-'),
    outcome: err ? 'failure' : 'success',
    errorCode: err ? String(err) : null,
    sessionId: str(record.insertId) || null,
    source: 'gcp',
    raw: record,
  });
}

/* ------------------------------------------------------------- CyberArk --- */

export function parseCyberArk(record) {
  const ts = toEpoch(
    record.timestamp || record.Time || record.time || record.EventTime || record.CreationTime,
  );
  if (Number.isNaN(ts)) return null;

  const op = str(
    record.action || record.Action || record.Operation || record.Desc || record.Message,
    'Activity',
  ).replace(/\s+/g, '');

  const durationSec = Number(record.session_duration_seconds ?? record.Duration ?? 0) || 0;
  const status = str(record.status || record.Status || record.Result, 'success');

  return makeEvent({
    ts,
    actor: str(record.user_id || record.User || record.Username || record.Issuer, 'unknown'),
    actorType: 'Human',
    action: `cyberark:${op}`,
    service: 'cyberark',
    resource: str(record.resource || record.Target || record.Safe || record.vault_name, '-'),
    ip: str(record.ip_address || record.SourceIP || record.ClientIP, '0.0.0.0'),
    region: str(record.region, '-'),
    userAgent: str(record.user_agent, 'CyberArk-Client'),
    outcome: /fail|denied|error/i.test(status) ? 'failure' : 'success',
    errorCode: /fail|denied|error/i.test(status) ? status : null,
    sessionId: str(record.session_id || record.SessionID) || null,
    source: 'cyberark',
    label: record.is_anomaly ? 1 : 0,
    raw: { ...record, durationSec },
  });
}

/* -------------------------------------------------------------- Generic --- */

/** Column aliases accepted by the generic parser, in priority order. */
const GENERIC_FIELDS = {
  ts: ['ts', 'timestamp', 'time', 'eventTime', 'datetime', 'date', '@timestamp', 'created_at'],
  actor: ['actor', 'user_id', 'user', 'username', 'principal', 'identity', 'account', 'upn', 'subject'],
  action: ['action', 'event', 'eventName', 'operation', 'activity', 'event_type', 'operationName'],
  resource: ['resource', 'target', 'object', 'asset', 'resource_name'],
  ip: ['ip', 'ip_address', 'sourceIPAddress', 'src_ip', 'client_ip', 'callerIpAddress'],
  region: ['region', 'location', 'awsRegion', 'datacenter', 'site'],
  country: ['country', 'geo_country', 'country_name'],
  userAgent: ['user_agent', 'userAgent', 'ua', 'client'],
  outcome: ['outcome', 'status', 'result', 'resultType', 'success'],
  sessionId: ['session_id', 'sessionId', 'correlationId', 'request_id'],
  label: ['label', 'is_anomaly', 'anomaly', 'ground_truth', 'malicious'],
};

function pick(record, names) {
  for (const n of names) {
    if (record[n] !== undefined && record[n] !== null && record[n] !== '') return record[n];
    const lower = Object.keys(record).find((k) => k.toLowerCase() === n.toLowerCase());
    if (lower && record[lower] !== undefined && record[lower] !== null && record[lower] !== '') {
      return record[lower];
    }
  }
  return undefined;
}

export function parseGeneric(record) {
  const ts = toEpoch(pick(record, GENERIC_FIELDS.ts));
  if (Number.isNaN(ts)) return null;

  let action = str(pick(record, GENERIC_FIELDS.action), 'unknown:Event');
  if (!action.includes(':')) {
    const svc = str(record.service || record.source || record.eventSource, 'log').replace(/\.amazonaws\.com$/, '');
    action = `${svc}:${action.replace(/\s+/g, '')}`;
  }

  const outcomeRaw = str(pick(record, GENERIC_FIELDS.outcome), 'success');
  const failed = /fail|error|denied|deny|false|4\d\d|5\d\d/i.test(outcomeRaw);
  const labelRaw = pick(record, GENERIC_FIELDS.label);

  return makeEvent({
    ts,
    actor: str(pick(record, GENERIC_FIELDS.actor), 'unknown'),
    action,
    resource: str(pick(record, GENERIC_FIELDS.resource), '-'),
    ip: str(pick(record, GENERIC_FIELDS.ip), '0.0.0.0'),
    region: str(pick(record, GENERIC_FIELDS.region), '-'),
    country: str(pick(record, GENERIC_FIELDS.country)) || null,
    userAgent: str(pick(record, GENERIC_FIELDS.userAgent), '-'),
    outcome: failed ? 'failure' : 'success',
    errorCode: failed ? outcomeRaw : null,
    sessionId: str(pick(record, GENERIC_FIELDS.sessionId)) || null,
    source: 'generic',
    label: labelRaw === true || labelRaw === 1 || labelRaw === '1' || labelRaw === 'true' ? 1 : 0,
    raw: record,
  });
}

/* --------------------------------------------------------------- Sniffer -- */

/**
 * Detect which parser a record set needs. Returns { format, parse, confidence }.
 * The decision is made from the first record that has any recognisable field,
 * not just index 0, because exports often start with a header-ish stub.
 */
export function detectFormat(records) {
  const sample = records.slice(0, 25);
  const has = (fn) => sample.some(fn);

  if (has((r) => r && r.eventVersion !== undefined && r.eventTime !== undefined)) {
    return { format: 'aws', parse: parseCloudTrail };
  }
  if (has((r) => r && r.eventSource && r.eventName && r.userIdentity)) {
    return { format: 'aws', parse: parseCloudTrail };
  }
  if (has((r) => r && r.userPrincipalName && (r.createdDateTime || r.appDisplayName))) {
    return { format: 'entra', parse: parseEntraSignIn };
  }
  if (has((r) => r && r.eventType && r.actor && r.client)) {
    return { format: 'okta', parse: parseOkta };
  }
  if (has((r) => r && r.protoPayload)) {
    return { format: 'gcp', parse: parseGcpAudit };
  }
  if (has((r) => r && r.operationName && (r.resourceId || r.correlationId))) {
    return { format: 'azure', parse: parseAzureActivity };
  }
  if (has((r) => r && (r.vault_name || r.Safe || r.privileged_account_used || r.PSMSessionID))) {
    return { format: 'cyberark', parse: parseCyberArk };
  }
  return { format: 'generic', parse: parseGeneric };
}

/** Pull the record array out of whatever wrapper the export used. */
export function unwrapRecords(data) {
  if (Array.isArray(data)) return data;
  if (!data || typeof data !== 'object') return [];
  for (const key of ['Records', 'records', 'value', 'events', 'items', 'data', 'logs', 'entries']) {
    if (Array.isArray(data[key])) return data[key];
  }
  return [data];
}

/** Minimal RFC4180-ish CSV reader (quoted fields, embedded commas/newlines). */
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else { quoted = false; }
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else if (c !== '\r') field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  if (!rows.length) return [];
  const header = rows[0].map((h) => h.trim());
  return rows.slice(1)
    .filter((r) => r.length && r.some((v) => v !== ''))
    .map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ''])));
}

/**
 * Ingest raw file text. Returns { events, format, skipped, total }.
 * Accepts JSON, JSONL/NDJSON and CSV; the shape decides the parser.
 */
export function ingest(text, hint = null) {
  const trimmed = text.trim();
  let records = [];

  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      records = unwrapRecords(JSON.parse(trimmed));
    } catch {
      // Not a single JSON document — try newline-delimited JSON.
      records = trimmed
        .split('\n')
        .map((line) => { try { return JSON.parse(line); } catch { return null; } })
        .filter(Boolean);
    }
  } else if (trimmed.includes(',') && trimmed.includes('\n')) {
    records = parseCsv(trimmed);
  } else {
    throw new Error('Unrecognised file: expected JSON, JSONL or CSV.');
  }

  if (!records.length) throw new Error('No records found in file.');

  const { format, parse } = hint && PARSERS[hint] ? { format: hint, parse: PARSERS[hint] } : detectFormat(records);

  const events = [];
  let skipped = 0;
  for (const r of records) {
    try {
      const e = parse(r);
      if (e) events.push(e); else skipped++;
    } catch {
      skipped++;
    }
  }
  events.sort((a, b) => a.ts - b.ts);
  return { events, format, skipped, total: records.length };
}

export const PARSERS = {
  aws: parseCloudTrail,
  azure: parseAzureActivity,
  entra: parseEntraSignIn,
  okta: parseOkta,
  gcp: parseGcpAudit,
  cyberark: parseCyberArk,
  generic: parseGeneric,
};

export const FORMAT_LABELS = {
  aws: 'AWS CloudTrail',
  azure: 'Azure Activity Log',
  entra: 'Microsoft Entra ID sign-ins',
  okta: 'Okta System Log',
  gcp: 'Google Cloud Audit Log',
  cyberark: 'CyberArk PAM',
  generic: 'Generic JSON/CSV',
  synthetic: 'ARGUS synthetic corpus',
};
