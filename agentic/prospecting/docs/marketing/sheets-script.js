/**
 * 1pont — Quiz answers → Google Sheet
 *
 * Setup:
 * 1. Create a new Google Sheet and note its ID from the URL.
 * 2. In the sheet, open Extensions → Apps Script.
 * 3. Paste this entire file, replacing the default content.
 * 4. Set SHEET_ID to your sheet's ID.
 * 5. Click Deploy → New deployment → Web app.
 *    - Execute as: Me
 *    - Who has access: Anyone
 * 6. Copy the deployment URL and paste it as SHEET_ENDPOINT in the landing pages.
 *
 * The script appends a row for every quiz completion and every waitlist signup.
 * Re-deploying after changes requires a new deployment (or "Manage deployments" → edit).
 */

const SHEET_ID = 'YOUR_GOOGLE_SHEET_ID_HERE';
const SECRET   = 'd891520657a0ab45f099a001465ad8d9dca866a4018401e2bde0971c9fdbef19';

const QUIZ_SHEET  = 'Quiz Answers';
const EMAIL_SHEET = 'Waitlist';

const QUIZ_HEADERS  = ['Timestamp','Name','Email','Phone','Score','Channels','Spend','Target','Data Quality','Agency','Frustration','Goal','Lang'];
const EMAIL_HEADERS = ['Timestamp','Email','Source'];

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);

    if (data.token !== SECRET) {
      return ContentService
        .createTextOutput(JSON.stringify({ ok: false, error: 'Forbidden' }))
        .setMimeType(ContentService.MimeType.JSON);
    }
    const ss   = SpreadsheetApp.openById(SHEET_ID);

    if (data.type === 'subscribe') {
      appendRow(ss, EMAIL_SHEET, EMAIL_HEADERS, [
        data.timestamp,
        data.email,
        data.source || 'landing',
      ]);
    } else {
      appendRow(ss, QUIZ_SHEET, QUIZ_HEADERS, [
        data.timestamp,
        data.name,
        data.email,
        data.phone,
        data.score,
        data.channels,
        data.spend,
        data.target,
        data.dataQuality,
        data.agency,
        data.frustration,
        data.goal,
        data.lang,
      ]);
    }

    return ContentService
      .createTextOutput(JSON.stringify({ ok: true }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: false, error: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function appendRow(ss, sheetName, headers, values) {
  let sheet = ss.getSheetByName(sheetName);
  if (!sheet) {
    sheet = ss.insertSheet(sheetName);
    sheet.appendRow(headers);
    sheet.getRange(1, 1, 1, headers.length)
         .setFontWeight('bold')
         .setBackground('#FF4D25')
         .setFontColor('#ffffff');
    sheet.setFrozenRows(1);
  }
  sheet.appendRow(values);
}

// Test manually from Apps Script editor: Run → doPost (pass dummy event)
function testPost() {
  doPost({
    postData: {
      contents: JSON.stringify({
        timestamp: new Date().toISOString(),
        name: 'Test User', email: 'test@example.com', phone: '',
        score: 42, channels: 'email, linkedin', spend: 'mid',
        target: 'b2b', dataQuality: 'ok', agency: 'solo',
        frustration: 'cost', goal: 'meetings', lang: 'en',
      }),
    },
  });
}
