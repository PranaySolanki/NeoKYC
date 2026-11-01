const fs = require('fs');
const path = require('path');
const { withDangerousMod, withMainApplication } = require('@expo/config-plugins');

const withRppgRoiMeans = (config) => {
  const packageName = config.android?.package;
  if (!packageName) throw new Error('withRppgRoiMeans requires expo.android.package');

  config = withMainApplication(config, (mod) => {
    let source = mod.modResults.contents;
    const importLine = `import ${packageName}.RppgRoiMeansPlugin`;
    if (!source.includes(importLine)) {
      source = source.replace(/^(package [^\r\n]+)(\r?\n)/, `$1$2\n${importLine}$2`);
    }
    if (!source.includes('RppgRoiMeansPlugin.register()')) {
      source = source.replace('loadReactNative(this)', 'RppgRoiMeansPlugin.register()\n    loadReactNative(this)');
    }
    mod.modResults.contents = source;
    return mod;
  });

  return withDangerousMod(config, ['android', async (mod) => {
    const sourcePath = path.join(__dirname, 'android', 'RppgRoiMeansPlugin.kt');
    const source = fs.readFileSync(sourcePath, 'utf8');
    const packagePath = packageName.replace(/\./g, path.sep);
    const targetDir = path.join(mod.modRequest.platformProjectRoot, 'app', 'src', 'main', 'java', packagePath);
    fs.mkdirSync(targetDir, { recursive: true });
    fs.writeFileSync(path.join(targetDir, 'RppgRoiMeansPlugin.kt'), `package ${packageName}\n\n${source}`);
    return mod;
  }]);
};

module.exports = withRppgRoiMeans;
