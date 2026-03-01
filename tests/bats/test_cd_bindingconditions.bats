# shellcheck disable=SC2148
# shellcheck disable=SC2329

setup_file() {
  load 'helpers.sh'
  local _iargs=("--set" "logVerbosity=6")
  iupgrade_wait "${TEST_CHART_REPO}" "${TEST_CHART_VERSION}" _iargs
}


# Executed before entering each test in this file.
setup() {
   load 'helpers.sh'
  _common_setup
  log_objects
}

bats::on_failure() {
  echo -e "\n\nFAILURE HOOK START"
  log_objects
  show_kubelet_plugin_error_logs
  show_kubelet_plugin_log_tails

  echo -e "\nCONTROLLER LOG TAILS START"
  (
    kubectl logs \
    -l nvidia-dra-driver-gpu-component=controller \
    -n nvidia-dra-driver-gpu \
    -c conpute-domain \
    --prefix --tail=400
  ) || true
  echo -e "KUBELET PLUGIN LOG TAILS END\n\n"

  echo -e "FAILURE HOOK END\n\n"
}

@test "CDs: ResourceSlice publishes BindingConditions for channel device" {
  local bc="ComputeDomainReady"
  local bfc="ComputeDomainNotReady"

  # Find a single ResourceSlice for this driver and assert it contains the BC/BFC names.
  local driver="compute-domain.nvidia.com"
  local rsname
  rsname="$(
    kubectl get resourceslice --no-headers \
      | awk -v d="${driver}" '$3==d {print $1; exit}'
  )"

  [[ -n "${rsname}" ]] || fail "no ResourceSlice found for driver=${driver}"

  # Get the ResourceSlice object
  run kubectl get resourceslice "${rsname}" -o json
  assert_success

  printf '%s' "${output}" | jq -e --arg bc "${bc}" --arg bfc "${bfc}" '
    .spec.devices[]
    | select(
      (.name | contains("channel"))
      and ((.bindingConditions // []) | index($bc))
      and ((.bindingFailureConditions // []) | index($bfc))
  ' > /dev/null || fail "ResourceSlice ${rsname} missing expected BindingConditions"
}

# Basic Flow
@test "CDs: workload pod waits for DS readiness before becoming Running" {
  local imex_demo_spec="demo/specs/imex/channel-injection.yaml"
  local imex_demo_cd_name="imex-channel-injection"
  local imex_demo_pod_name="imex-channel-injection"
  local imex_demo_rct_name="imex-channel-0"

  # Create ComputeDomain + Pod
  kubectl apply -f "${imex_demo_spec}"
  kubectl wait --for=condition=READY pods "${imex_demo_pod_name}" --timeout=80s

  local claim=$(kubectl get resourceclaim | grep "${imex_demo_pod_name}" | awk '{print $1}')

  [[ -n "${claim}" ]] || fail "Could not find ResourceClaim owned by pod ${imex_demo_pod_name}"

  retry 60 1 kubectl get resourceclaim "${claim}" -o jsonpath='{.status.allocation.devices.results[0].device}' \
    || fail "claim did not become allocated"

  local node
  node="$(kubectl get resourceclaim "${claim}" -o jsonpath='{.status.allocation.nodeSelector.nodeSelectorTerms[0].matchFields[0].values[0]}' 2>/dev/null || true)"
  [[ -n "${node}" ]] || fail "allocated node empty"

  local label
  label=$(kubectl get node "${node}" -o jsonpath='{.metadata.labels["resource.nvidia.com/computeDomain"]}' 2>/dev/null || true)
}