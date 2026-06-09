#include "aceimprovise/server/protocol.hpp"

#include "yyjson.h"

#include <cstring>

namespace aceimprovise {

namespace {

const char * encode_status_str(EncodeStatus status) {
    switch (status) {
    case EncodeStatus::Ready:
        return "ready";
    case EncodeStatus::Pending:
        return "pending";
    case EncodeStatus::Failed:
        return "failed";
    }
    return "ready";
}

void write_prompts(yyjson_mut_doc * doc, yyjson_mut_val * root, const std::vector<PromptEntry> & prompts) {
    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    for (const PromptEntry & p : prompts) {
        yyjson_mut_val * obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, obj, "id", p.id.c_str());
        yyjson_mut_obj_add_str(doc, obj, "caption", p.caption.c_str());
        yyjson_mut_obj_add_real(doc, obj, "weight", p.weight);
        yyjson_mut_obj_add_bool(doc, obj, "muted", p.muted);
        yyjson_mut_obj_add_str(doc, obj, "encode_status", encode_status_str(p.encode_status));
        yyjson_mut_arr_append(arr, obj);
    }
    yyjson_mut_obj_add_val(doc, root, "prompts", arr);
}

}  // namespace

std::string session_to_json(const SessionView & view) {
    yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_str(doc, root, "liveset_name", view.liveset_name.c_str());
    yyjson_mut_obj_add_bool(doc, root, "playing", view.playing);
    yyjson_mut_obj_add_str(doc, root, "composed_caption", view.composed_caption.c_str());
    yyjson_mut_obj_add_int(doc, root, "active_prompt_count", view.active_prompt_count);
    yyjson_mut_obj_add_bool(doc, root, "inference_ready", view.inference_ready);
    yyjson_mut_obj_add_bool(doc, root, "generating", view.generating);
    if (!view.inference_error.empty()) {
        yyjson_mut_obj_add_str(doc, root, "inference_error", view.inference_error.c_str());
    }
    yyjson_mut_obj_add_int(doc, root, "prompt_count", static_cast<int>(view.prompts.size()));

    write_prompts(doc, root, view.prompts);

    yyjson_mut_val * shared = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_real(doc, shared, "denoise", view.denoise);
    yyjson_mut_obj_add_real(doc, shared, "cover_strength", view.cover_strength);
    yyjson_mut_obj_add_real(doc, shared, "guidance", view.guidance);
    yyjson_mut_obj_add_real(doc, shared, "feedback", view.feedback);
    yyjson_mut_obj_add_int(doc, shared, "feedback_depth", view.feedback_depth);
    yyjson_mut_obj_add_val(doc, root, "shared", shared);

    yyjson_mut_val * pipeline = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_int(doc, pipeline, "depth", view.pipeline.depth);
    yyjson_mut_obj_add_int(doc, pipeline, "steps", view.pipeline.steps);
    yyjson_mut_obj_add_bool(doc, pipeline, "streaming", view.pipeline.streaming);
    yyjson_mut_obj_add_int(doc, pipeline, "tick_count", static_cast<int>(view.pipeline.tick_count));
    yyjson_mut_obj_add_int(doc, pipeline, "finished_latents", static_cast<int>(view.pipeline.finished_latents));
    yyjson_mut_obj_add_int(doc, pipeline, "active_slots", view.pipeline.active_slots);
    yyjson_mut_obj_add_int(doc, pipeline, "queue_depth", view.pipeline.queue_depth);
    yyjson_mut_obj_add_real(doc, pipeline, "last_tick_ms", view.pipeline.last_tick_ms);
    yyjson_mut_obj_add_val(doc, root, "pipeline", pipeline);

    const char * json = yyjson_mut_write(doc, YYJSON_WRITE_NOFLAG, nullptr);
    std::string out   = json ? json : "{}";
    if (json) {
        free((void *) json);
    }
    yyjson_mut_doc_free(doc);
    return out;
}

std::string error_json(const std::string & message) {
    yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "error", message.c_str());
    const char * json = yyjson_mut_write(doc, YYJSON_WRITE_NOFLAG, nullptr);
    std::string out   = json ? json : "{}";
    if (json) {
        free((void *) json);
    }
    yyjson_mut_doc_free(doc);
    return out;
}

}  // namespace aceimprovise
