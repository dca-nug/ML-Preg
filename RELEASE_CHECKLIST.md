# Release checklist

Zenodo only archives a GitHub release created **after** the repository toggle
is switched on. A release created before the toggle is not picked up, and
switching the toggle afterwards does not archive it retroactively. The fix is
to delete the release and the tag, then recreate them. Steps 1 to 3 exist to
avoid that.

## 1. Link the account

- Zenodo → Account → GitHub → Connect.
- Grant access to the correct GitHub account or organisation.

## 2. Toggle the correct repository

- Find this repository in the Zenodo GitHub list.
- Switch the toggle **on** for this repository, not for a neighbouring one.
- Reload the page and confirm the toggle is still on before continuing.

## 3. Confirm the webhook exists

- GitHub → repository → Settings → Webhooks.
- A `zenodo.org` webhook should be listed with a green tick.
- Absent webhook means the toggle did not take. Go back to step 2.

## 4. Prepare the repository

- [ ] `.zenodo.json` present, valid JSON, creators and affiliations correct
- [ ] `CITATION.cff` version matches the tag you are about to create
- [ ] `CHANGELOG.md` has an entry for this version with a real date
- [ ] `LICENSE` year and name correct
- [ ] `README.md` has no remaining `TODO` markers except the DOI badge
- [ ] `git status` clean, no data file staged
- [ ] `git ls-files | grep -iE '\.(pkl|csv)$'` returns nothing unexpected

## 5. Create the release

```bash
git tag -a v1.0.0 -m "v1.0.0"
git push origin v1.0.0
```

Then create the GitHub release from that tag. Zenodo archives on the release
event, not on the tag push.

## 6. Verify

- [ ] Zenodo shows a new record within a few minutes
- [ ] The record has both a version DOI and a concept DOI
- [ ] Metadata matches `.zenodo.json`
- [ ] Files in the archive contain no data and no `.pkl`

## 7. Backfill

- [ ] Add the DOI badge to `README.md`
- [ ] Add the concept DOI to `CITATION.cff`
- [ ] Commit both, then create a `v1.0.1` release if you want the archived
      copy to contain its own DOI

The concept DOI resolves to the latest version and is the one to cite in the
manuscript. A version DOI pins one release and is the one to cite if a
specific run needs to be identified.
