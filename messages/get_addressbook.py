#!/usr/local/bin/python

from collections import defaultdict
import functools
from pprint import pprint
import sqlite3
import sys
from pathlib import Path
import json
import re
import glob

def main(input_dir: str, output_file: str):

    db_files = resolve_db_files(input_dir)

    if not db_files:
        print('No .abcddb files found.')
        sys.exit(1)

    print('Processing %d database file(s):' % len(db_files))
    for f in db_files:
        print('  - %s' % f)

    phone_to_name = {}
    total_people_printed = 0

    for dbfile in db_files:
        # sqlite requires only the .abcddb path; wal/shm are auto-used if present in same dir
        if not Path(dbfile).exists():
            print('SKIP: file not found: %s' % dbfile)
            continue

        con = sqlite3.connect(dbfile)
        con.row_factory = sqlite3.Row

        with con:
            cur = con.cursor()

            contact_index = {}

            cur.execute(recordSql)
            rows = cur.fetchall()
            for row in rows:
                cid = row['Z_PK']
                d = dict(row)
                d = filterNoneValues(d)
                if d:
                    contact_index[cid] = filterNoneValues(d)

            addElements(contact_index, emailSql, cur, 'ZOWNER', 'email')
            addElements(contact_index, mailSql, cur, 'ZOWNER', 'mail')
            addElements(contact_index, phoneSql, cur, 'ZOWNER', 'phone')
            addElements(contact_index, imSql, cur, 'ZOWNER', 'im')
            addElements(contact_index, noteSql, cur, 'ZCONTACT', 'note')

            contacts = list(contact_index.values())
            contacts.sort(key=functools.cmp_to_key(cmpContacts))

            people = []
            ddl = lambda: defaultdict(list)

            for c in contacts:
                person = {'locations': defaultdict(ddl)}

                nickname = c.get('ZNICKNAME')
                if nickname:
                    nickname = '"%s"' % nickname
                maidenname = c.get('MAIDENNAME')
                if maidenname:
                    maidenname = '"%s"' % maidenname
                name = [c.get('ZTITLE'), c.get('ZFIRSTNAME'), nickname, c.get('ZMIDDLENAME'), c.get('ZLASTNAME'), maidenname, c.get('ZSUFFIX')]
                name = [x for x in name if x]
                if not name:
                    continue
                person['name'] = ' '.join(name)
                people.append(person)

                email = c.get('email')
                if email:
                    for e in email:
                        if e.get('ZADDRESS'):
                            person['locations'][e.get('label')]['email'].append(e.get('ZADDRESS'))

                mail = c.get('mail')
                if mail:
                    for m in mail:
                        addr = [m.get('street'), m.get('ZCITY'), m.get('ZSTATE'), m.get('ZZIPCODE'), m.get('ZCOUNTRYNAME'), m.get('ZREGION'), m.get('ZSAMA')]
                        addr = [_f for _f in addr if _f]
                        person['locations'][m.get('label')]['mail'].append(', '.join(addr))

                phone = c.get('phone')
                if phone:
                    for p in phone:
                        # keep aggregated print view
                        num = [p.get('ZAREACODE'), p.get('ZCOUNTRYCODE'), p.get('ZEXTENSION'), p.get('ZLOCALNUMBER'), p.get('fullnumber')]
                        num = [_f for _f in num if _f]
                        person['locations'][p.get('label')]['phone'].append(', '.join(num))

                        # Save EVERY number for this contact across ALL databases
                        preferred = (p.get('fullnumber') or '').strip()
                        if not preferred:
                            pieces = ''.join([x for x in [p.get('ZCOUNTRYCODE'), p.get('ZAREACODE'), p.get('ZLOCALNUMBER'), p.get('ZEXTENSION')] if x])
                            preferred = pieces
                        normalized_phone = normalize_phone(preferred, p.get('ZCOUNTRYCODE'))
                        if normalized_phone:
                            phone_to_name[normalized_phone] = person['name']

                im = c.get('im')
                if im:
                    for e in im:
                        if e.get('ZADDRESS'):
                            if not person.get('IM'):
                                person['IM'] = []
                            person['IM'].append('%s (%s)' % (e.get('ZADDRESS'), e.get('ZSERVICENAME')))

                notes = c.get('note')
                if notes:
                    for e in notes:
                        if e.get('text'):
                            if not person.get('notes'):
                                person['notes'] = []
                            person['notes'].append(e.get('text'))

            # Existing printing (kept)
            for p in people:
                print(p['name'])
                locs = list(p['locations'].items())
                if len(locs) > 1:
                    for k, v in list(p['locations'].items()):
                        print('    %s' % k)
                        for x in v['mail']:
                            print('        %s' % x)
                        for x in v['phone']:
                            print('        %s' % x)
                        for x in v['email']:
                            print('        %s' % x)
                elif len(locs) == 1:
                    _, v = locs[0]
                    for x in v['mail']:
                        print('    %s' % x)
                    for x in v['phone']:
                        print('    %s' % x)
                    for x in v['email']:
                        print('    %s' % x)

                im = p.get('IM')
                if im:
                    print('    IM')
                    for x in im:
                        print('        %s' % x)

                notes = p.get('notes')
                if notes:
                    print('    notes')
                    for x in notes:
                        print('        %s' % x)

            total_people_printed += len(people)

    # Write merged mapping
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(phone_to_name, f, ensure_ascii=False, indent=2)
        print('SUCCESS: wrote %d phone→name entries from %d db file(s) to %s' %
              (len(phone_to_name), len(db_files), output_file))
        print('Printed details for %d contacts total.' % total_people_printed)
    except Exception as e:
        print('ERROR: failed to write %s: %s' % (output_file, e))


def resolve_db_files(arg):
    files = []

    p = Path(arg)

    # directory contains multiple {folder_uuid}/AddressBook-v22.abcddb, read them all
    files = list(p.glob('**/*.abcddb'))
    if not files:
        print('No .abcddb files found in directory: %s' % p)
        return []

    # remove files starting with ._
    files = [f for f in files if not Path(f).name.startswith('._')]

    # Filter to unique, existing
    uniq = []
    seen = set()
    for f in files:
        if f not in seen and Path(f).exists():
            seen.add(f)
            uniq.append(f)
    return uniq


def normalize_phone(s, country_code=None):
    """Normalize a phone number to digits, keep + if present,
    and add +country_code if missing."""
    s = s.strip()
    if not s:
        return ''

    if s.startswith('+'):
        return '+' + re.sub(r'\D+', '', s[1:])

    digits = re.sub(r'\D+', '', s)

    if country_code:
        cc = re.sub(r'\D+', '', country_code)
        return f"+{cc}{digits}"
    else:
        return f"+1{digits}"


def filterNoneValues(d):
    return dict((k, v) for k, v in d.items() if v is not None)

def addElements(contact_index, sql, cur, id_field, name):
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        cid = row[id_field]
        if not contact_index[cid].get(name):
            contact_index[cid][name] = []
        d = filterNoneValues(dict(row))
        del d[id_field]
        if d:
            contact_index[cid][name].append(d)

def cmpContacts(a, b):
    def _cmp(x, y):
        return (x > y) - (x < y)

    a_key_1 = a.get('ZFIRSTNAME') or a.get('ZLASTNAME') or a.get('ZORGANIZATION') or ''
    a_key_2 = a.get('ZLASTNAME') or a_key_1
    b_key_1 = b.get('ZFIRSTNAME') or b.get('ZLASTNAME') or b.get('ZORGANIZATION') or ''
    b_key_2 = b.get('ZLASTNAME') or b_key_1

    return _cmp(a_key_1, b_key_1) or _cmp(a_key_2, b_key_2)


recordSql = '''
    select
        r.Z_PK,
        date(r.ZBIRTHDAY, 'unixepoch', '+31 years') as birthday,  /* Mac dates relative to 2000? */
        r.ZTITLE, r.ZFIRSTNAME, r.ZNICKNAME, r.ZMIDDLENAME, r.ZLASTNAME, r.ZSUFFIX, r.ZMAIDENNAME,
        r.ZORGANIZATION, r.ZJOBTITLE, r.ZTMPHOMEPAGE
    from ZABCDRECORD r;
'''

emailSql = '''
    select
        e.ZOWNER, lower(replace(replace(e.ZLABEL, '_$!<', ''), '>!$_', '')) as label, e.ZADDRESS
    from ZABCDEMAILADDRESS e;
'''

mailSql = '''
    select
        a.ZOWNER, lower(replace(replace(a.ZLABEL, '_$!<', ''), '>!$_', '')) as label, trim(replace(a.ZSTREET, char(10), ', ')) as street, a.ZCITY, a.ZSTATE, a.ZZIPCODE, a.ZCOUNTRYCODE, a.ZCOUNTRYNAME, a.ZREGION, a.ZSAMA
    from ZABCDPOSTALADDRESS a;
'''

phoneSql = '''
    select
        p.ZOWNER, lower(replace(replace(p.ZLABEL, '_$!<', ''), '>!$_', '')) as label, p.ZAREACODE, p.ZCOUNTRYCODE, p.ZEXTENSION, p.ZLOCALNUMBER, trim(replace(p.ZFULLNUMBER, char(10), ', ')) as fullnumber
    from ZABCDPHONENUMBER p;
'''

imSql = '''
    select
        m.ZOWNER, s.ZSERVICENAME, m.ZADDRESS
    from ZABCDMESSAGINGADDRESS m
    left outer join ZABCDSERVICE s on m.ZSERVICE = s.Z_PK;
'''

noteSql = '''
    select
        n.ZCONTACT, trim(replace(n.ZTEXT, char(10), ', ')) as text
    from ZABCDNOTE n;
'''

if __name__ == '__main__':
    # parse args
    import argparse
    parser = argparse.ArgumentParser(description="Extract contact information from AddressBook database files.")
    parser.add_argument("data_dir", type=str, help="Directory containing 'Sources/' folder.")
    args = parser.parse_args()
    
    input_dir = Path(args.data_dir).expanduser() / "Sources"
    output_file = Path(args.data_dir).expanduser() / "phone_to_name.json"

    main(input_dir, output_file)