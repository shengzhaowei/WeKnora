package service

import (
	"context"
	"errors"
	"testing"

	"github.com/Tencent/WeKnora/internal/models/embedding"
	"github.com/Tencent/WeKnora/internal/types"
	"github.com/Tencent/WeKnora/internal/types/interfaces"
	"github.com/stretchr/testify/require"
)

// recordingRetrieveEngine is a stateful fake vector store: BatchIndex embeds
// through the embedder (like the real engines) and stores rows per chunk,
// DeleteByChunkIDList drops the rows of the given chunks.
type recordingRetrieveEngine struct {
	interfaces.RetrieveEngineService
	indexed         map[string][]*types.IndexInfo
	deleteCalls     int
	batchIndexCalls int
}

func (e *recordingRetrieveEngine) EngineType() types.RetrieverEngineType {
	return types.PostgresRetrieverEngineType
}

func (e *recordingRetrieveEngine) Support() []types.RetrieverType {
	return []types.RetrieverType{types.VectorRetrieverType}
}

func (e *recordingRetrieveEngine) EstimateStorageSize(
	context.Context, embedding.Embedder, []*types.IndexInfo, []types.RetrieverType,
) int64 {
	return 0
}

func (e *recordingRetrieveEngine) BatchIndex(
	ctx context.Context,
	embedder embedding.Embedder,
	infos []*types.IndexInfo,
	_ []types.RetrieverType,
) error {
	e.batchIndexCalls++
	texts := make([]string, len(infos))
	for i, info := range infos {
		texts[i] = info.Content
	}
	if _, err := embedder.BatchEmbed(ctx, texts); err != nil {
		return err
	}
	for _, info := range infos {
		e.indexed[info.ChunkID] = append(e.indexed[info.ChunkID], info)
	}
	return nil
}

func (e *recordingRetrieveEngine) DeleteByChunkIDList(
	_ context.Context, ids []string, _ int, _ string,
) error {
	e.deleteCalls++
	for _, id := range ids {
		delete(e.indexed, id)
	}
	return nil
}

func (e *recordingRetrieveEngine) rowCount() int {
	n := 0
	for _, rows := range e.indexed {
		n += len(rows)
	}
	return n
}

// failingEmbedder simulates an embedding service that is temporarily
// offline: every embedding call fails.
type failingEmbedder struct {
	parentChildEmbedder
}

func (failingEmbedder) Embed(context.Context, string) ([]float32, error) {
	return nil, errors.New("embedding service unavailable")
}

func (failingEmbedder) BatchEmbed(context.Context, []string) ([][]float32, error) {
	return nil, errors.New("embedding service unavailable")
}

func appendSkipFixture(t *testing.T) (*faqWriteFixture, *recordingRetrieveEngine) {
	t.Helper()
	f := newFAQWriteFixture(t)
	sqlDB, err := f.db.DB()
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, sqlDB.Close()) })
	require.NoError(t, f.db.Where("id = ?", "one").Delete(&types.Chunk{}).Error)
	engine := &recordingRetrieveEngine{indexed: map[string][]*types.IndexInfo{}}
	f.svc.retrieveEngine = &fakeFanoutRegistry{
		byStore: map[string]interfaces.RetrieveEngineService{*f.kb.VectorStoreID: engine},
	}
	f.svc.modelService = parentChildModelService{embedder: parentChildEmbedder{}}
	f.svc.chunkService = &chunkService{
		chunkRepository: f.chunks, knowledgeRepo: f.svc.repo,
		kbRepository: &imageReaderKBRepo{kb: f.kb},
	}
	return f, engine
}

func eightFAQEntries() []types.FAQEntryPayload {
	enabled, disabled := true, false
	return []types.FAQEntryPayload{
		{
			StandardQuestion: "如何重置密码？", SimilarQuestions: []string{"密码忘了怎么办"},
			Answers: []string{"在登录页点忘记密码。"}, TagID: 11, IsEnabled: &enabled,
		},
		{
			StandardQuestion: "如何申请请假？", SimilarQuestions: []string{"请假流程是什么", "年假怎么请"},
			Answers: []string{"在 OA 提交请假申请。"}, TagID: 11, IsEnabled: &enabled,
		},
		{
			StandardQuestion: "报销多久到账？",
			Answers:          []string{"审批通过后 3 个工作日到账。"}, TagID: 11, IsEnabled: &enabled,
		},
		{
			StandardQuestion: "如何开通 VPN？", SimilarQuestions: []string{"VPN 怎么申请"},
			NegativeQuestions: []string{"如何关闭 VPN"},
			Answers:           []string{"找 IT 开通 VPN 账号。"}, TagID: 11, IsEnabled: &enabled,
		},
		{
			StandardQuestion: "会议室怎么预定？",
			Answers:          []string{"在会议系统里预定。"}, TagID: 11, IsEnabled: &disabled,
		},
		{
			StandardQuestion: "加班餐补标准？",
			Answers:          []string{"加班满 2 小时补 25 元。", "凭加班记录领取。"}, TagID: 11, IsEnabled: &enabled,
		},
		{
			StandardQuestion: "入职需要哪些材料？", SimilarQuestions: []string{"新员工入职材料"},
			Answers: []string{"身份证、学历证、体检报告。"}, TagID: 11, IsEnabled: &enabled,
		},
		{
			StandardQuestion: "如何修改手机号？",
			Answers:          []string{"在 HR 系统里修改。"}, TagID: 11, IsEnabled: &enabled,
		},
	}
}

func runAppendImport(
	t *testing.T, f *faqWriteFixture, taskID string, entries []types.FAQEntryPayload,
) (*types.FAQImportProgress, error) {
	t.Helper()
	progress := &types.FAQImportProgress{TaskID: taskID, KBID: f.kb.ID}
	err := f.svc.executeFAQImport(f.ctx, taskID, f.kb.ID,
		&types.FAQBatchUpsertPayload{Mode: types.FAQBatchModeAppend, Entries: entries}, f.kb.TenantID, 0, progress)
	return progress, err
}

// faqChunkVersions returns standard question -> metadata version for every
// FAQ chunk of the fixture knowledge base.
func faqChunkVersions(t *testing.T, f *faqWriteFixture) map[string]int {
	t.Helper()
	var chunks []types.Chunk
	require.NoError(t, f.db.Where("knowledge_id = ?", "faq").Find(&chunks).Error)
	versions := make(map[string]int, len(chunks))
	for _, chunk := range chunks {
		meta, err := chunk.FAQMetadata()
		require.NoError(t, err)
		require.NotNil(t, meta)
		versions[meta.StandardQuestion] = meta.Version
	}
	return versions
}

func TestFAQAppendImportSkipsUnchangedEntries(t *testing.T) {
	f, engine := appendSkipFixture(t)
	entries := eightFAQEntries()

	progress, err := runAppendImport(t, f, "append-skip-1", entries)
	require.NoError(t, err)
	require.Len(t, progress.SuccessEntries, 8)
	require.Equal(t, map[string]int{
		"如何重置密码？": 1, "如何申请请假？": 1, "报销多久到账？": 1, "如何开通 VPN？": 1,
		"会议室怎么预定？": 1, "加班餐补标准？": 1, "入职需要哪些材料？": 1, "如何修改手机号？": 1,
	}, faqChunkVersions(t, f))
	require.Equal(t, 8, engine.rowCount())
	require.Equal(t, 0, engine.deleteCalls)
	batchIndexCalls := engine.batchIndexCalls

	// Re-importing the identical batch in append mode must skip everything:
	// no merges, no version bump, no index rebuild.
	progress, err = runAppendImport(t, f, "append-skip-2", eightFAQEntries())
	require.NoError(t, err)
	require.Equal(t, 0, progress.MergedCount)
	require.Equal(t, map[string]int{
		"如何重置密码？": 1, "如何申请请假？": 1, "报销多久到账？": 1, "如何开通 VPN？": 1,
		"会议室怎么预定？": 1, "加班餐补标准？": 1, "入职需要哪些材料？": 1, "如何修改手机号？": 1,
	}, faqChunkVersions(t, f))
	require.Equal(t, 8, engine.rowCount())
	require.Equal(t, 0, engine.deleteCalls)
	require.Equal(t, batchIndexCalls, engine.batchIndexCalls)
}

func TestFAQAppendImportStillMergesChangedEntries(t *testing.T) {
	f, engine := appendSkipFixture(t)

	progress, err := runAppendImport(t, f, "append-merge-1", eightFAQEntries())
	require.NoError(t, err)
	require.Len(t, progress.SuccessEntries, 8)

	changed := eightFAQEntries()
	changed[2].Answers = []string{"审批通过后 5 个工作日到账。"}
	progress, err = runAppendImport(t, f, "append-merge-2", changed)
	require.NoError(t, err)
	require.Equal(t, 1, progress.MergedCount)
	versions := faqChunkVersions(t, f)
	require.Equal(t, 2, versions["报销多久到账？"])
	for q, v := range versions {
		if q != "报销多久到账？" {
			require.Equal(t, 1, v, "unchanged entry %q must stay at version 1", q)
		}
	}
	require.Equal(t, 8, engine.rowCount())

	var chunk types.Chunk
	require.NoError(t, f.db.Where("knowledge_id = ?", "faq").First(&chunk,
		"json_extract(metadata, '$.standard_question') = ?", "报销多久到账？").Error)
	meta, err := chunk.FAQMetadata()
	require.NoError(t, err)
	require.Equal(t, []string{"审批通过后 5 个工作日到账。"}, meta.Answers)
}

func TestFAQAppendImportKeepsIndexWhenEmbeddingOffline(t *testing.T) {
	f, engine := appendSkipFixture(t)

	progress, err := runAppendImport(t, f, "append-offline-1", eightFAQEntries())
	require.NoError(t, err)
	require.Len(t, progress.SuccessEntries, 8)
	require.Equal(t, 8, engine.rowCount())

	// Embedding service goes offline.
	f.svc.modelService = parentChildModelService{embedder: failingEmbedder{}}

	// Re-importing unchanged entries while offline must skip them entirely:
	// no merge, no version bump, index rows untouched, import succeeds.
	progress, err = runAppendImport(t, f, "append-offline-2", eightFAQEntries())
	require.NoError(t, err)
	require.Equal(t, 0, progress.MergedCount)
	require.Equal(t, 8, engine.rowCount())
	require.Equal(t, 0, engine.deleteCalls)
	for _, v := range faqChunkVersions(t, f) {
		require.Equal(t, 1, v)
	}

	// A genuinely changed entry imported while offline must fail WITHOUT
	// deleting the existing index rows, and must not bump the PG version,
	// so a later retry can still see the change and rebuild the index.
	changed := eightFAQEntries()
	changed[2].Answers = []string{"审批通过后 5 个工作日到账。"}
	_, err = runAppendImport(t, f, "append-offline-3", changed)
	require.Error(t, err)
	require.Equal(t, 8, engine.rowCount())
	require.Equal(t, 0, engine.deleteCalls)
	versions := faqChunkVersions(t, f)
	require.Equal(t, 1, versions["报销多久到账？"])

	// Embedding service recovers: the retry applies the change.
	f.svc.modelService = parentChildModelService{embedder: parentChildEmbedder{}}
	progress, err = runAppendImport(t, f, "append-offline-4", changed)
	require.NoError(t, err)
	require.Equal(t, 1, progress.MergedCount)
	require.Equal(t, 2, faqChunkVersions(t, f)["报销多久到账？"])
	require.Equal(t, 8, engine.rowCount())
}
