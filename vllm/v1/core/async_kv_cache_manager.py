    def schedule(self) -> SchedulerOutput:
        with self.schedule_condition:
            self.schedule_condition.wait_for(
                lambda: self.preschedule_step > self.schedule_step)
        if self.schedule_step % 2 == 0:
            output = self.even_output
        else : 
            output = self.odd_output            
        
        # # Delete request finished in previous round
        for finish_req_id in self.finished_req_ids:
            if finish_req_id in output.num_scheduled_tokens:
                output.scheduled_cached_reqs = list(
                    filter(lambda item: item.req_id != \
                           finish_req_id, output.scheduled_cached_reqs))
                output.total_num_scheduled_tokens -= \
                    output.num_scheduled_tokens.get(finish_req_id)
                output.num_scheduled_tokens.pop(finish_req_id, None)
                output.scheduled_spec_decode_tokens.pop(finish_req_id, None)
                output.scheduled_encoder_inputs.pop(finish_req_id, None)
                output.finished_req_ids.add(finish_req_id)
                output.structured_output_request_ids.pop(finish_req_id, None)
                # TODO: grammar_bitmask
        
            self.pre_num_computed_tokens.pop(finish_req_id, None)
        for req_data in output.scheduled_new_reqs:
            req = self.requests[req_data.req_id]
            self.pre_num_computed_tokens[req_data.req_id] = req.num_computed_tokens
            self.kv_cache_manager.cache_full_block(req)

        # Modify request data according to newly generated tokens 
        # in previous round.
        for req_data in output.scheduled_cached_reqs:
            req = self.requests[req_data.req_id]
            pre_num_computed_tokens = self.pre_num_computed_tokens[
                req_data.req_id]
            num_computed_tokens = req.num_computed_tokens
            new_token_ids = req.all_token_ids[
                pre_num_computed_tokens:num_computed_tokens]
            req_data.new_token_ids = new_token_ids
            self.pre_num_computed_tokens[req_data.req_id] = num_computed_tokens
            self.kv_cache_manager.cache_full_block(req)

        if len(self.running) > 0:
            output.num_common_prefix_blocks = \
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    self.running[0], len(self.running))
        else:
            output.num_common_prefix_blocks = 0
        self.finished_req_ids = set()
        
        with self.preschedule_condition:
            self.schedule_step += 1
            self.preschedule_condition.notify()
        if len(output.num_scheduled_tokens) == 0:
            grammer_bitmask = output.grammar_bitmask
            output = SchedulerOutput(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=[],
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=0,
                finished_req_ids=self.finished_req_ids,
                free_encoder_input_ids=[],
                structured_output_request_ids={},
                grammar_bitmask=grammer_bitmask
            )
        return output