def prepare() {
  // Checkout the parent repository. 
  checkout([
     $class: 'GitSCM',
     branches:  [[name: '*/master']],
     doGenerateSubmoduleConfigurations: false,
     extensions: [[$class: 'SubmoduleOption',
                   disableSubmodules: false,
                   parentCredentials: true,
                   recursiveSubmodules: true,
                   reference: '',
                   trackingSubmodules: false]],
     submoduleCfg: [],
     userRemoteConfigs: [[credentialsId: 'github', url: 'https://github.com/ciphermodelabs/ciphercore_private']],
  ])
  // Checkout our branch to the corresponding subdirectory.
  dir('public') {
    checkout scm
  }
}

pipeline {
  agent none
  options {
    skipDefaultCheckout true
  }
  stages {
    stage('CI') {
      failFast true
      parallel {
         stage('Code Formatting') {
           agent any
           steps {
             prepare()
             dir('private') {
               sh './run_code_formatting_docker.sh'
             }
           }
         }
         stage('Clippy') {
           agent any
           steps {
             prepare()
             dir('private') {
               sh './run_clippy_docker.sh'
             }
           }
         }
         stage('Tests') {
           agent any
           steps {
             prepare()
             dir('private') {
               sh './run_tests_docker.sh'
             }
           }
         }
         stage('Docs') {
           agent any
           steps {
             prepare()
             dir('private') {
               sh './run_docs_docker.sh'
               publishHTML(target: [
                 allowMissing: false,
                 alwaysLinkToLastBuild: true,
                 keepAll: true,
                 reportDir: 'doc',
                 reportFiles: 'ciphercore_utils/index.html,ciphercore_base/index.html,ciphercore_evaluators/index.html,ciphercore_runtime/index.html',
                 reportName: 'Documentation'])
             }
           }
         }
         stage('Coverage') {
           agent any
           steps {
             prepare()
             dir('private') {
               sh './run_coverage_docker.sh'
               publishHTML(target: [
                 allowMissing: false,
                 alwaysLinkToLastBuild: true,
                 keepAll: true,
                 reportDir: 'coverage_output',
                 reportFiles: 'coverage_summary_ciphercore-utils.html,coverage_report_ciphercore-utils.html,coverage_summary_ciphercore-base.html,coverage_report_ciphercore-base.html,coverage_summary_ciphercore-evaluators.html,coverage_report_ciphercore-evaluators.html,coverage_summary_ciphercore-runtime.html,coverage_report_ciphercore-runtime.html',
                 reportName: 'Code Coverage'])
             }
           }
         }
         // These should be re-enabled once performance issues are resolved.
         //stage('Address Sanitizer') {
         //  agent any
         //  steps {
         //    prepare()
         //    dir('private') {
         //      sh './run_address_sanitizer_docker.sh'
         //    }
         //  }
         //}
         //stage('Thread Sanitizer') {
         //  agent any
         //  steps {
         //    prepare()
         //    dir('private') {
         //      sh './run_thread_sanitizer_docker.sh'
         //    }
         //  }
         //}
       } 
    }
    stage('Benchmarks') {
      agent any
      steps {
        prepare()
        dir('private') {
          sh './run_benchmarks_docker.sh'
        }
      }
    }
  }
}
