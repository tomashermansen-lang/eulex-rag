"""Tests for cross-law suite manager.

TDD: These tests verify cross_law_suite_manager.py provides correct CRUD,
validation, and YAML import/export for cross-law eval suites.

Requirement mapping:
- CSM-001: list_suites returns all suites (R6.1)
- CSM-002: get_suite returns specific suite (R6.5)
- CSM-003: create_suite saves to YAML (R6.5)
- CSM-004: update_suite modifies existing (R8.1)
- CSM-005: delete_suite removes file (R8.3)
- CSM-006: import_yaml parses correctly (R6.3)
- CSM-007: import_yaml rejects malformed YAML (E16)
- CSM-008: export_yaml produces valid YAML (R6.4)
- CSM-009: Validates duplicate case IDs (E13)
- CSM-010: Validates corpus IDs exist (E14)
- CSM-011: Validates comparison has 2+ corpora (E15)
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.cross_law_suite_manager import (
    CrossLawSuiteManager,
    CrossLawEvalSuite,
    CrossLawGoldenCase,
    SuiteValidationError,
)


@pytest.fixture
def temp_evals_dir():
    """Create a temporary directory for test suites."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_corpus_registry():
    """Return a set of valid corpus IDs for validation."""
    return {"ai-act", "gdpr", "nis2", "dora", "cra"}


@pytest.fixture
def manager(temp_evals_dir, mock_corpus_registry):
    """Create a suite manager with temp directory."""
    return CrossLawSuiteManager(
        evals_dir=temp_evals_dir,
        valid_corpus_ids=mock_corpus_registry,
    )


@pytest.fixture
def sample_case():
    """Create a sample cross-law golden case."""
    return CrossLawGoldenCase(
        id="compare_ai_gdpr_transparency",
        prompt="Compare AI-Act and GDPR on transparency requirements",
        corpus_scope="explicit",
        target_corpora=("ai-act", "gdpr"),
        synthesis_mode="comparison",
        expected_anchors=("AI-Act Art. 13", "GDPR Art. 12"),
        expected_corpora=("ai-act", "gdpr"),
        min_corpora_cited=2,
        profile="LEGAL",
        disabled=False,
        origin="manual",
    )


@pytest.fixture
def sample_suite(sample_case):
    """Create a sample cross-law eval suite."""
    return CrossLawEvalSuite(
        id="transparency_comparison_suite",
        name="Transparency Comparison Tests",
        description="Tests comparing transparency requirements across laws",
        target_corpora=("ai-act", "gdpr"),
        default_synthesis_mode="comparison",
        cases=(sample_case,),
    )


class TestCrossLawSuiteManagerCRUD:
    """Tests for CRUD operations."""

    def test_csm_001_list_suites_returns_all_suites(self, manager, sample_suite):
        """list_suites should return all cross-law suites."""
        # Create 3 suites
        for i in range(3):
            suite = CrossLawEvalSuite(
                id=f"suite_{i}",
                name=f"Suite {i}",
                description="",
                target_corpora=("ai-act", "gdpr"),
                default_synthesis_mode="comparison",
                cases=(),
            )
            manager.create_suite(suite)

        # List should return all 3
        suites = manager.list_suites()
        assert len(suites) == 3
        assert {s.id for s in suites} == {"suite_0", "suite_1", "suite_2"}

    def test_csm_002_get_suite_returns_specific_suite(self, manager, sample_suite):
        """get_suite should return specific suite by ID."""
        manager.create_suite(sample_suite)

        result = manager.get_suite(sample_suite.id)

        assert result is not None
        assert result.id == sample_suite.id
        assert result.name == sample_suite.name
        assert len(result.cases) == 1
        assert result.cases[0].id == "compare_ai_gdpr_transparency"

    def test_csm_002b_get_suite_returns_none_for_unknown(self, manager):
        """get_suite should return None for unknown ID."""
        result = manager.get_suite("nonexistent_suite")
        assert result is None

    def test_csm_003_create_suite_saves_to_yaml(self, manager, sample_suite, temp_evals_dir):
        """create_suite should persist suite as YAML file."""
        manager.create_suite(sample_suite)

        # Check file exists
        expected_file = temp_evals_dir / f"cross_law_{sample_suite.id}.yaml"
        assert expected_file.exists()

        # Verify content is valid YAML
        import yaml
        with open(expected_file) as f:
            data = yaml.safe_load(f)
        assert data["id"] == sample_suite.id
        assert data["name"] == sample_suite.name

    def test_csm_004_update_suite_modifies_existing(self, manager, sample_suite):
        """update_suite should modify existing suite."""
        manager.create_suite(sample_suite)

        # Modify the suite
        updated = CrossLawEvalSuite(
            id=sample_suite.id,
            name="Updated Name",
            description="Updated description",
            target_corpora=sample_suite.target_corpora,
            default_synthesis_mode=sample_suite.default_synthesis_mode,
            cases=sample_suite.cases,
        )
        manager.update_suite(updated)

        # Retrieve and verify
        result = manager.get_suite(sample_suite.id)
        assert result.name == "Updated Name"
        assert result.description == "Updated description"

    def test_csm_004b_update_nonexistent_raises_error(self, manager):
        """update_suite should raise error for nonexistent suite."""
        fake_suite = CrossLawEvalSuite(
            id="nonexistent",
            name="Fake",
            description="",
            target_corpora=("ai-act",),
            default_synthesis_mode="unified",
            cases=(),
        )
        with pytest.raises(FileNotFoundError):
            manager.update_suite(fake_suite)

    def test_csm_005_delete_suite_removes_file(self, manager, sample_suite, temp_evals_dir):
        """delete_suite should remove the YAML file."""
        manager.create_suite(sample_suite)

        expected_file = temp_evals_dir / f"cross_law_{sample_suite.id}.yaml"
        assert expected_file.exists()

        manager.delete_suite(sample_suite.id)

        assert not expected_file.exists()
        assert manager.get_suite(sample_suite.id) is None

    def test_csm_005b_delete_nonexistent_raises_error(self, manager):
        """delete_suite should raise error for nonexistent suite."""
        with pytest.raises(FileNotFoundError):
            manager.delete_suite("nonexistent")


class TestCrossLawSuiteManagerYAML:
    """Tests for YAML import/export."""

    def test_csm_006_import_yaml_parses_correctly(self, manager):
        """import_yaml should parse valid YAML string into suite."""
        yaml_content = """
id: imported_suite
name: Imported Suite
description: A test suite
target_corpora:
  - ai-act
  - gdpr
default_synthesis_mode: comparison
cases:
  - id: case_1
    prompt: Compare AI-Act and GDPR
    corpus_scope: explicit
    target_corpora:
      - ai-act
      - gdpr
    synthesis_mode: comparison
    expected_anchors: []
    expected_corpora:
      - ai-act
      - gdpr
    min_corpora_cited: 2
    profile: LEGAL
    disabled: false
    origin: manual
"""
        suite = manager.import_yaml(yaml_content)

        assert suite.id == "imported_suite"
        assert suite.name == "Imported Suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "case_1"

    def test_csm_007_import_yaml_rejects_malformed(self, manager):
        """import_yaml should reject malformed YAML with error."""
        malformed_yaml = """
id: test
name: Test
cases:
  - id: case_1
    prompt: Missing closing bracket
    target_corpora: [ai-act, gdpr
"""
        with pytest.raises(SuiteValidationError) as exc_info:
            manager.import_yaml(malformed_yaml)

        assert "YAML" in str(exc_info.value) or "parse" in str(exc_info.value).lower()

    def test_csm_008_export_yaml_produces_valid_yaml(self, manager, sample_suite):
        """export_yaml should produce CLI-compatible YAML."""
        manager.create_suite(sample_suite)

        yaml_str = manager.export_yaml(sample_suite.id)

        # Should be valid YAML
        import yaml
        data = yaml.safe_load(yaml_str)
        assert data["id"] == sample_suite.id
        assert data["name"] == sample_suite.name
        assert len(data["cases"]) == 1

    def test_csm_008b_export_roundtrip(self, manager, sample_suite):
        """Export then import should preserve data."""
        manager.create_suite(sample_suite)
        yaml_str = manager.export_yaml(sample_suite.id)

        # Create new manager with different temp dir
        with tempfile.TemporaryDirectory() as tmpdir2:
            manager2 = CrossLawSuiteManager(
                evals_dir=Path(tmpdir2),
                valid_corpus_ids={"ai-act", "gdpr", "nis2", "dora", "cra"},
            )
            reimported = manager2.import_yaml(yaml_str)

            assert reimported.id == sample_suite.id
            assert reimported.name == sample_suite.name
            assert len(reimported.cases) == len(sample_suite.cases)


class TestCrossLawSuiteManagerValidation:
    """Tests for suite validation."""

    def test_csm_009_validates_duplicate_case_ids(self, manager):
        """Suite with duplicate case IDs should be rejected."""
        case1 = CrossLawGoldenCase(
            id="duplicate_id",
            prompt="Question 1",
            corpus_scope="all",
            target_corpora=(),
            synthesis_mode="unified",
            expected_anchors=(),
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        case2 = CrossLawGoldenCase(
            id="duplicate_id",  # Same ID!
            prompt="Question 2",
            corpus_scope="all",
            target_corpora=(),
            synthesis_mode="unified",
            expected_anchors=(),
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        suite = CrossLawEvalSuite(
            id="suite_with_dupes",
            name="Suite",
            description="",
            target_corpora=("ai-act", "gdpr"),
            default_synthesis_mode="unified",
            cases=(case1, case2),
        )

        with pytest.raises(SuiteValidationError) as exc_info:
            manager.create_suite(suite)

        assert "duplicate" in str(exc_info.value).lower()

    def test_csm_010_validates_corpus_ids_exist(self, manager):
        """Suite with unknown corpus_id should be rejected."""
        case = CrossLawGoldenCase(
            id="invalid_corpus_case",
            prompt="Question",
            corpus_scope="explicit",
            target_corpora=("ai-act", "unknown_law"),  # Invalid!
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("ai-act", "unknown_law"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        suite = CrossLawEvalSuite(
            id="suite_bad_corpus",
            name="Suite",
            description="",
            target_corpora=("ai-act", "unknown_law"),
            default_synthesis_mode="comparison",
            cases=(case,),
        )

        with pytest.raises(SuiteValidationError) as exc_info:
            manager.create_suite(suite)

        assert "unknown_law" in str(exc_info.value)

    def test_csm_011_validates_comparison_has_2_plus_corpora(self, manager):
        """Comparison mode with <2 corpora should be rejected."""
        case = CrossLawGoldenCase(
            id="single_corpus_comparison",
            prompt="Compare?",
            corpus_scope="explicit",
            target_corpora=("ai-act",),  # Only 1!
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("ai-act",),
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        suite = CrossLawEvalSuite(
            id="suite_bad_comparison",
            name="Suite",
            description="",
            target_corpora=("ai-act",),
            default_synthesis_mode="comparison",
            cases=(case,),
        )

        with pytest.raises(SuiteValidationError) as exc_info:
            manager.create_suite(suite)

        assert "comparison" in str(exc_info.value).lower()
        assert "2" in str(exc_info.value)

    def test_csm_validates_empty_prompt_rejected(self, manager):
        """Case with empty prompt should be rejected."""
        case = CrossLawGoldenCase(
            id="empty_prompt_case",
            prompt="",  # Empty!
            corpus_scope="all",
            target_corpora=(),
            synthesis_mode="unified",
            expected_anchors=(),
            expected_corpora=("ai-act",),
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        suite = CrossLawEvalSuite(
            id="suite_empty_prompt",
            name="Suite",
            description="",
            target_corpora=("ai-act", "gdpr"),
            default_synthesis_mode="unified",
            cases=(case,),
        )

        with pytest.raises(SuiteValidationError) as exc_info:
            manager.create_suite(suite)

        assert "prompt" in str(exc_info.value).lower()


class TestCrossLawSuiteManagerCaseOperations:
    """Tests for case-level operations."""

    def test_add_case_to_suite(self, manager, sample_suite, sample_case):
        """Adding a case to suite should update the suite."""
        manager.create_suite(sample_suite)

        new_case = CrossLawGoldenCase(
            id="new_case",
            prompt="New question",
            corpus_scope="all",
            target_corpora=(),
            synthesis_mode="aggregation",
            expected_anchors=(),
            expected_corpora=("ai-act", "gdpr", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        manager.add_case(sample_suite.id, new_case)

        result = manager.get_suite(sample_suite.id)
        assert len(result.cases) == 2
        assert any(c.id == "new_case" for c in result.cases)

    def test_update_case_in_suite(self, manager, sample_suite):
        """Updating a case in suite should persist change."""
        manager.create_suite(sample_suite)
        original_case_id = sample_suite.cases[0].id

        updated_case = CrossLawGoldenCase(
            id=original_case_id,
            prompt="Updated prompt text",
            corpus_scope="explicit",
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=("AI-Act Art. 13",),  # Changed
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        manager.update_case(sample_suite.id, updated_case)

        result = manager.get_suite(sample_suite.id)
        case = next(c for c in result.cases if c.id == original_case_id)
        assert case.prompt == "Updated prompt text"
        assert case.expected_anchors == ("AI-Act Art. 13",)

    def test_delete_case_from_suite(self, manager, sample_suite):
        """Deleting a case from suite should remove it."""
        manager.create_suite(sample_suite)
        case_id = sample_suite.cases[0].id

        manager.delete_case(sample_suite.id, case_id)

        result = manager.get_suite(sample_suite.id)
        assert len(result.cases) == 0

    def test_duplicate_case(self, manager, sample_suite):
        """Duplicating a case should create copy with new ID."""
        manager.create_suite(sample_suite)
        original_case_id = sample_suite.cases[0].id

        new_id = manager.duplicate_case(sample_suite.id, original_case_id)

        result = manager.get_suite(sample_suite.id)
        assert len(result.cases) == 2
        assert any(c.id == original_case_id for c in result.cases)
        assert any(c.id == new_id for c in result.cases)
        assert new_id != original_case_id


class TestCrossLawGoldenCaseDataclass:
    """Tests for the CrossLawGoldenCase dataclass."""

    def test_case_is_frozen(self):
        """CrossLawGoldenCase should be immutable."""
        case = CrossLawGoldenCase(
            id="test",
            prompt="Test",
            corpus_scope="all",
            target_corpora=(),
            synthesis_mode="unified",
            expected_anchors=(),
            expected_corpora=(),
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        with pytest.raises(AttributeError):
            case.prompt = "Modified"

    def test_suite_has_timestamps(self, manager, sample_suite, temp_evals_dir):
        """Suite should have created_at and modified_at timestamps."""
        manager.create_suite(sample_suite)
        result = manager.get_suite(sample_suite.id)

        assert result.created_at is not None
        assert result.modified_at is not None


class TestCrossLawGoldenCaseExpandedFields:
    """Tests for expanded CrossLawGoldenCase fields (C1: R-ED-02 to R-ED-11)."""

    def test_t1_1_instantiate_with_all_new_fields(self):
        """T1.1: Can instantiate CrossLawGoldenCase with all new fields."""
        case = CrossLawGoldenCase(
            id="full_case",
            prompt="Compare AI-Act and GDPR on transparency",
            corpus_scope="explicit",
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=("ai-act:article:13",),
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
            # New fields
            test_types=("corpus_coverage", "comparison_completeness"),
            expected_behavior="answer",
            must_include_any_of=("ai-act:article:13",),
            must_include_any_of_2=(),
            must_include_all_of=("gdpr:article:6",),
            must_not_include_any_of=("nis2:article:1",),
            contract_check=True,
            min_citations=2,
            max_citations=10,
            notes="Test note for transparency comparison",
        )

        assert case.test_types == ("corpus_coverage", "comparison_completeness")
        assert case.expected_behavior == "answer"
        assert case.must_include_any_of == ("ai-act:article:13",)
        assert case.must_include_all_of == ("gdpr:article:6",)
        assert case.must_not_include_any_of == ("nis2:article:1",)
        assert case.contract_check is True
        assert case.min_citations == 2
        assert case.max_citations == 10
        assert case.notes == "Test note for transparency comparison"

    def test_t1_2_defaults_work_original_fields_only(self):
        """T1.2: All new fields have defaults — instantiate with only original fields."""
        case = CrossLawGoldenCase(
            id="minimal",
            prompt="Question",
            corpus_scope="explicit",
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("ai-act",),
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )

        assert case.test_types == ()
        assert case.expected_behavior == "answer"
        assert case.must_include_any_of == ()
        assert case.must_include_any_of_2 == ()
        assert case.must_include_all_of == ()
        assert case.must_not_include_any_of == ()
        assert case.contract_check is False
        assert case.min_citations is None
        assert case.max_citations is None
        assert case.notes == ""

    def test_t1_3_yaml_roundtrip_with_new_fields(self, manager):
        """T1.3: YAML save + load preserves all new fields."""
        case = CrossLawGoldenCase(
            id="roundtrip_case",
            prompt="Roundtrip test question for cross-law",
            corpus_scope="explicit",
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=2,
            profile="ENGINEERING",
            disabled=True,
            origin="auto-generated",
            test_types=("corpus_coverage", "synthesis_balance"),
            expected_behavior="abstain",
            must_include_any_of=("ai-act:article:13", "gdpr:article:12"),
            must_include_any_of_2=("ai-act:article:14",),
            must_include_all_of=("gdpr:article:6",),
            must_not_include_any_of=("nis2:article:1",),
            contract_check=True,
            min_citations=3,
            max_citations=8,
            notes="Important roundtrip test",
        )
        suite = CrossLawEvalSuite(
            id="roundtrip_suite",
            name="Roundtrip Test",
            description="",
            target_corpora=("ai-act", "gdpr"),
            default_synthesis_mode="comparison",
            cases=(case,),
        )
        manager.create_suite(suite)

        loaded = manager.get_suite("roundtrip_suite")
        c = loaded.cases[0]

        assert c.test_types == ("corpus_coverage", "synthesis_balance")
        assert c.expected_behavior == "abstain"
        assert c.must_include_any_of == ("ai-act:article:13", "gdpr:article:12")
        assert c.must_include_any_of_2 == ("ai-act:article:14",)
        assert c.must_include_all_of == ("gdpr:article:6",)
        assert c.must_not_include_any_of == ("nis2:article:1",)
        assert c.contract_check is True
        assert c.min_citations == 3
        assert c.max_citations == 8
        assert c.notes == "Important roundtrip test"

    def test_t1_4_yaml_load_old_format_defaults(self, manager):
        """T1.4: Loading YAML without new fields applies defaults."""
        old_yaml = """
id: old_suite
name: Old Format Suite
description: Created before new fields existed
target_corpora:
  - ai-act
  - gdpr
default_synthesis_mode: comparison
cases:
  - id: old_case
    prompt: Old question without new fields
    corpus_scope: explicit
    target_corpora:
      - ai-act
      - gdpr
    synthesis_mode: comparison
    expected_anchors:
      - AI-Act Art. 13
    expected_corpora:
      - ai-act
      - gdpr
    min_corpora_cited: 2
    profile: LEGAL
    disabled: false
    origin: auto-generated
"""
        suite = manager.import_yaml(old_yaml)
        c = suite.cases[0]

        # All new fields should have defaults
        assert c.test_types == ()
        assert c.expected_behavior == "answer"
        assert c.must_include_any_of == ()
        assert c.must_include_any_of_2 == ()
        assert c.must_include_all_of == ()
        assert c.must_not_include_any_of == ()
        assert c.contract_check is False
        assert c.min_citations is None
        assert c.max_citations is None
        assert c.notes == ""
        # Original fields still loaded
        assert c.expected_anchors == ("AI-Act Art. 13",)
        assert c.profile == "LEGAL"


class TestCrossLawGoldenCaseQualityFields:
    """Tests for discovery, difficulty, and answerability fields (R1.1, R2.5, R3.1)."""

    def test_ce_001_discovery_accepted_as_synthesis_mode(self, manager):
        """CE-001: 'discovery' is accepted as valid synthesis_mode."""
        case = CrossLawGoldenCase(
            id="discovery_case",
            prompt="Hvad kræver EU-lovgivning om ICT-risikostyring?",
            corpus_scope="all",
            target_corpora=("dora", "nis2"),
            synthesis_mode="discovery",
            expected_anchors=(),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )
        suite = CrossLawEvalSuite(
            id="discovery_suite",
            name="Discovery Test",
            description="",
            target_corpora=("dora", "nis2"),
            default_synthesis_mode="discovery",
            cases=(case,),
        )
        # Should not raise
        manager.create_suite(suite)
        loaded = manager.get_suite("discovery_suite")
        assert loaded.cases[0].synthesis_mode == "discovery"

    def test_ce_002_difficulty_field(self):
        """CE-002: difficulty field accepts 'easy'/'medium'/'hard'/None."""
        for difficulty in ("easy", "medium", "hard", None):
            case = CrossLawGoldenCase(
                id=f"diff_{difficulty}",
                prompt="Q",
                corpus_scope="all",
                target_corpora=(),
                synthesis_mode="unified",
                expected_anchors=(),
                expected_corpora=(),
                min_corpora_cited=1,
                profile="LEGAL",
                disabled=False,
                origin="manual",
                difficulty=difficulty,
            )
            assert case.difficulty == difficulty

    def test_ce_005_yaml_roundtrip_new_quality_fields(self, manager):
        """CE-005: YAML round-trip preserves difficulty."""
        case = CrossLawGoldenCase(
            id="quality_roundtrip",
            prompt="Compare DORA and NIS2 on ICT risk",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            difficulty="hard",
        )
        suite = CrossLawEvalSuite(
            id="quality_roundtrip_suite",
            name="Quality Roundtrip",
            description="",
            target_corpora=("dora", "nis2"),
            default_synthesis_mode="comparison",
            cases=(case,),
        )
        manager.create_suite(suite)
        loaded = manager.get_suite("quality_roundtrip_suite")
        c = loaded.cases[0]

        assert c.difficulty == "hard"

    def test_ce_006_backward_compat_yaml_without_new_fields(self, manager):
        """CE-006: YAML without new quality fields loads with defaults."""
        old_yaml = """
id: compat_suite
name: Compat
description: ""
target_corpora:
  - dora
  - nis2
default_synthesis_mode: comparison
cases:
  - id: old_case
    prompt: Old question
    corpus_scope: explicit
    target_corpora:
      - dora
      - nis2
    synthesis_mode: comparison
    expected_anchors: []
    expected_corpora:
      - dora
      - nis2
    min_corpora_cited: 2
    profile: LEGAL
    disabled: false
    origin: manual
"""
        suite = manager.import_yaml(old_yaml)
        c = suite.cases[0]

        assert c.difficulty is None

    def test_ce_007_expected_corpora_subset_of_target_corpora(self, manager):
        """CE-007: expected_corpora must be subset of target_corpora."""
        case = CrossLawGoldenCase(
            id="subset_fail",
            prompt="Question",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("dora", "cra"),  # cra not in target_corpora!
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        suite = CrossLawEvalSuite(
            id="subset_fail_suite",
            name="Subset Fail",
            description="",
            target_corpora=("dora", "nis2"),
            default_synthesis_mode="comparison",
            cases=(case,),
        )

        with pytest.raises(SuiteValidationError) as exc_info:
            manager.create_suite(suite)
        assert "cra" in str(exc_info.value).lower() or "subset" in str(exc_info.value).lower()

    def test_ce_008_duplicate_case_copies_new_fields(self, manager):
        """CE-008: duplicate_case copies difficulty."""
        case = CrossLawGoldenCase(
            id="original_quality",
            prompt="Original question",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            difficulty="medium",
        )
        suite = CrossLawEvalSuite(
            id="dup_quality_suite",
            name="Dup Quality",
            description="",
            target_corpora=("dora", "nis2"),
            default_synthesis_mode="comparison",
            cases=(case,),
        )
        manager.create_suite(suite)

        new_id = manager.duplicate_case("dup_quality_suite", "original_quality")
        loaded = manager.get_suite("dup_quality_suite")
        duped = next(c for c in loaded.cases if c.id == new_id)

        assert duped.difficulty == "medium"


# =========================================================================
# Phase 1: Tests for min_corpora_cited fix and retrieval_confirmed field
# =========================================================================


class TestMinCorporaCitedNullable:
    """Tests for min_corpora_cited accepting None (type fix)."""

    def test_min_corpora_cited_accepts_none(self):
        """CrossLawGoldenCase should accept min_corpora_cited=None."""
        case = CrossLawGoldenCase(
            id="disc_test",
            prompt="Topic question",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="discovery",
            expected_anchors=(),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=None,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )
        assert case.min_corpora_cited is None

    def test_min_corpora_cited_still_accepts_int(self):
        """Existing int values still work."""
        case = CrossLawGoldenCase(
            id="int_test",
            prompt="Compare",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        assert case.min_corpora_cited == 2

    def test_min_corpora_cited_none_yaml_roundtrip(self, manager):
        """min_corpora_cited=None survives YAML round-trip."""
        case = CrossLawGoldenCase(
            id="null_mcc",
            prompt="Topic question",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="discovery",
            expected_anchors=(),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=None,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )
        suite = CrossLawEvalSuite(
            id="mcc_test_suite",
            name="MCC Test",
            description="",
            target_corpora=("dora", "nis2"),
            default_synthesis_mode="discovery",
            cases=(case,),
        )
        manager.create_suite(suite)
        loaded = manager.get_suite("mcc_test_suite")
        assert loaded.cases[0].min_corpora_cited is None


class TestRetrievalConfirmedField:
    """Tests for retrieval_confirmed field on CrossLawGoldenCase."""

    def test_retrieval_confirmed_default_none(self):
        """retrieval_confirmed should default to None."""
        case = CrossLawGoldenCase(
            id="rc_test",
            prompt="Test",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        assert case.retrieval_confirmed is None

    def test_retrieval_confirmed_accepts_true(self):
        """retrieval_confirmed can be set to True."""
        case = CrossLawGoldenCase(
            id="rc_true",
            prompt="Test",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=("article:6",),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            retrieval_confirmed=True,
        )
        assert case.retrieval_confirmed is True

    def test_retrieval_confirmed_accepts_false(self):
        """retrieval_confirmed can be set to False."""
        case = CrossLawGoldenCase(
            id="rc_false",
            prompt="Test",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=("article:6",),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            retrieval_confirmed=False,
        )
        assert case.retrieval_confirmed is False

    def test_retrieval_confirmed_yaml_roundtrip(self, manager):
        """retrieval_confirmed survives YAML round-trip."""
        case = CrossLawGoldenCase(
            id="rc_yaml",
            prompt="Test prompt",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=("article:6",),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            retrieval_confirmed=True,
        )
        suite = CrossLawEvalSuite(
            id="rc_yaml_suite",
            name="RC YAML",
            description="",
            target_corpora=("dora", "nis2"),
            default_synthesis_mode="comparison",
            cases=(case,),
        )
        manager.create_suite(suite)
        loaded = manager.get_suite("rc_yaml_suite")
        assert loaded.cases[0].retrieval_confirmed is True

    def test_backward_compat_no_retrieval_confirmed(self, manager):
        """Existing YAML without retrieval_confirmed loads as None."""
        case = CrossLawGoldenCase(
            id="old_case",
            prompt="Old prompt",
            corpus_scope="explicit",
            target_corpora=("dora", "nis2"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("dora", "nis2"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        suite = CrossLawEvalSuite(
            id="old_suite",
            name="Old Suite",
            description="",
            target_corpora=("dora", "nis2"),
            default_synthesis_mode="comparison",
            cases=(case,),
        )
        manager.create_suite(suite)
        loaded = manager.get_suite("old_suite")
        # Should default to None for cases created without the field
        assert loaded.cases[0].retrieval_confirmed is None
